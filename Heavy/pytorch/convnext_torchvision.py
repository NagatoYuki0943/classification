import torch
from torch import nn
from torchvision import models

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 224, 224).to(device)

model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
model.to(device)

for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False
    else:
        print(name)
        # classifier.0.weight
        # classifier.0.bias
        # classifier.2.weight
        # classifier.2.bias

params = [param for param in model.parameters() if param.requires_grad]

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 10]

# 预处理方式
transform = models.ConvNeXt_Tiny_Weights.DEFAULT.transforms()
print(transform)
# ImageClassification(
#     crop_size=[224]
#     resize_size=[236]
#     mean=[0.485, 0.456, 0.406]
#     std=[0.229, 0.224, 0.225]
#     interpolation=InterpolationMode.BILINEAR
# )
