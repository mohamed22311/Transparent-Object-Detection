import torch

def make_anchors(x, strides, offset=0.5):
    """Generate anchors from features.
    Args:
        x (List[Tensor]): List of feature maps.
        strides (List[int]): List of strides.
        offset (float): Grid cell offset.
    Returns:
        anchor_points (Tensor): Anchor points.
        stride_tensor (Tensor): Stride tensor.
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)