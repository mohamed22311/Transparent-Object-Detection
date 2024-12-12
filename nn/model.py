import torch.nn as nn
from .blocks import Conv, fuse_conv
from .backbone import Backbone
from .head import Head
from .neck import Neck

class YOLO(nn.Module):
    def __init__(self, num_classes: int, base_channels: int, base_depth: int, deep_mul: float):
        super().__init__()
        self.backbone = Backbone(base_channels, base_depth, deep_mul)
        self.neck = Neck(base_channels, base_depth, deep_mul)


        width = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.initialize_biases()

        #img_dummy = torch.zeros(1, 3, 256, 256)
        #self.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self