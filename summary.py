#--------------------------------------------#
#   This script is used to inspect the network architecture
#--------------------------------------------#
import torch
from thop import clever_format, profile

from nn import BaseModel

if __name__ == "__main__":
    # Define input parameters
    input_shape = [640, 640]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 80
    phi = 's'

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = BaseModel(input_shape, num_classes, phi, False).to(device)

    # Print model layers
    for layer in model.children():
        print(layer)
        print('==============================')

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)

    # Calculate FLOPs and parameters
    flops, params = profile(model, (dummy_input,), verbose=False)

    # Adjust FLOPs calculation
    # flops * 2 because profile does not count convolution as two operations
    # Some papers consider convolution as both multiplication and addition (x2)
    # Others only consider multiplication (x1)
    # This code multiplies by 2, following the approach in YOLOX
    flops *= 2

    # Format FLOPs and parameters
    flops, params = clever_format([flops, params], "%.3f")

    # Print results
    print(f'Total GFLOPS: {flops}')
    print(f'Total params: {params}')