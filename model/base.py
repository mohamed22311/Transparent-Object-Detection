import torch.nn as nn
from .blocks import Conv, fuse_conv
from .backbone import Backbone
from .neck import Neck
from .head import Head

class BaseModel(nn.Module):
    def __init__(self, num_classes: int, base_channels: int, base_depth: int, deep_mul: float):
        super().__init__()
        self.backbone = Backbone(base_channels, base_depth, deep_mul)
        self.neck = Neck(base_channels, base_depth, deep_mul)

        # Define the width of the feature maps for the head
        width = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.head = Head(num_classes, (width[0], width[1], width[2]))
        self.head.initialize_biases()

    def forward(self, x):
        # Extract features from the backbone
        x = self.backbone(x)
        # Process features in the neck
        x = self.neck(x)
        # Perform detection in the head
        return self.head(list(x))

    def fuse(self):
        # Fuse Conv and BatchNorm layers for optimization
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self