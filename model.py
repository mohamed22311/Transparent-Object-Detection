import torch
from nn import BaseModel

class Model:
    def __init__(self, phi: str, num_classes: int, input_channels: int, width_multiplier: float, depth_multiplier: float):

        
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3

        self.model = BaseModel(num_classes, base_channels, base_depth, deep_mul)
