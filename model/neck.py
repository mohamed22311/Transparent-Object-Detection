import torch
import torch.nn as nn
from .blocks import Conv, CBAM, TransformerEncoderBlock

class Neck(nn.Module):
    """ Model Neck."""
    def __init__(self, base_channels, base_depth, deep_mul):
        """Initializes the Model neck.
        Args:
            base_channels (int): Number of base channels.
            base_depth (int): Number of Bottleneck blocks in the first three layers.
            deep_mul (int): Multiplier for the number of channels in the last layer.
        """

        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.h1 = nn.Sequential(
            TransformerEncoderBlock(channels=int(base_channels * 16 * deep_mul) + base_channels * 8),
            CBAM(base_channels * 8)  # Add CBAM for attention
        )
        
        # 768, 80, 80 => 256, 80, 80
        self.h2 = nn.Sequential(
            TransformerEncoderBlock(channels=base_channels * 8 + base_channels * 4),
            CBAM(base_channels * 4)  # Add CBAM for attention
        )
        
        # 256, 80, 80 => 256, 40, 40
        self.h3 = Conv(c1=base_channels * 4,
                       c2=base_channels * 4,
                       k=3,
                       s=2)
        
        # 512 + 256, 40, 40 => 512, 40, 40
        self.h4 = nn.Sequential(
            TransformerEncoderBlock(channels=base_channels * 8 + base_channels * 4),
            CBAM(base_channels * 8)  # Add CBAM for attention
        )

        # 512, 40, 40 => 512, 20, 20
        self.h5 = Conv(c1=base_channels * 8,
                       c2=base_channels * 8,
                       k=3,
                       s=2)
        
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.h6 = nn.Sequential(
            TransformerEncoderBlock(channels=int(base_channels * 16 * deep_mul) + base_channels * 8),
            CBAM(int(base_channels * 16 * deep_mul))  # Add CBAM for attention
        )

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6
        # h1: p4
        # h2: p3
        # h4: p4
        # h6: p5