import torch
from torch import nn
from torchvision.models import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 224, 224).to(device)
model = vgg11_bn().to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 1000]
