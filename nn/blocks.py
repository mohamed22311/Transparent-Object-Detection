import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  
    """Pad to 'same' shape outputs.
    Args:
        k (int or tuple): Kernel size.
        p (int or tuple): Padding size.
        d (int or tuple): Dilation size.
    Returns:
        Padding size.
        """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation.
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int or tuple): Kernel size.
            s (int): Stride.
            p (int or tuple): Padding.
            g (int): Number of groups.
            d (int): Dilation.
            act (bool or nn.Module): Activation layer.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=c1,
                              out_channels=c2,
                              kernel_size=k,
                              stride=s,
                              padding=autopad(k, p, d),
                              groups=g,
                              dilation=d,
                              bias=False)
        
        self.norm = nn.BatchNorm2d(num_features=c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))
    
class Bottleneck(nn.Module):
    """Standard bottleneck. 1x1, 3x3, 1x1 convolution with shortcut connection."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters.
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool): Whether to include a shortcut connection.
            g (int): Number of groups for grouped convolution.
            k (int or tuple): Kernel size for the 3x3 convolution.
            e (float): Bottleneck expansion factor.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing.
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to include a shortcut connection.
            g (int): Number of groups for grouped convolution.
            e (float): Bottleneck expansion factor.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1)) # split input
        y.extend(m(y[-1]) for m in self.m) # apply n Bottleneck blocks
        return self.cv2(torch.cat(y, 1)) # concatenate and apply conv2

    def forward_split(self, x):
        """Forward pass using split() instead of chunk().
        (split() is faster but may consume more memory.)"""
        y = self.cv1(x).split((self.c, self.c), 1) # split input
        y = [y[0], y[1]] 
        y.extend(m(y[-1]) for m in self.m) # apply n Bottleneck blocks
        return self.cv2(torch.cat(y, 1)) # concatenate and apply conv2
    
class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fusion layer."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1) # input convolution
        self.cv2 = Conv(c_ * 4, c2, 1, 1) # output convolution
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) # maxpool layer

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)] # input convolution 
        y.extend(self.m(y[-1]) for _ in range(3)) # apply maxpool 3 times
        return self.cv2(torch.cat(y, 1)) # concatenate and apply output convolution


class DFL(torch.nn.Module):
    """Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391"""
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)

def fuse_conv(conv: nn.Conv2d, norm: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuses a Conv2d layer and a BatchNorm2d layer into a single Conv2d layer.
    Args:
        conv (torch.nn.Conv2d): Conv2d layer.
        norm (torch.nn.BatchNorm2d): BatchNorm2d layer.
    Returns:
        torch.nn.Conv2d: Fused Conv2d layer.

    The procedure is described in https://tehnokv.com/posts/fusing-batchnorm-and-conv/.
    Breifly, the weights of the Conv2d layer are modified to include the normalization parameters.
    """
    fused_conv = nn.Conv2d(in_channels=conv.in_channels,
                           out_channels=conv.out_channels,
                           kernel_size=conv.kernel_size,
                           stride=conv.stride,
                           padding=conv.padding,
                           groups=conv.groups,
                           bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv