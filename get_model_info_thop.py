import torch
from torchvision import models
from thop import profile


if __name__ == "__main__":
    model = models.resnet18()
    x = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs = {str(flops / 1000 ** 3)} G")
    print(f"Total params: {params}")
    # FLOPs = 1.824033792 G
    # Total params: 11689512.0
