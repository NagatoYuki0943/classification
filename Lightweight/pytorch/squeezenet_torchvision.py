"""
Squeeze没使用fc,只使用了conv,所以要修改conv层
"""

import torch
from torch.nn import Linear, Conv2d
from torchvision.models import squeezenet1_1


device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 224, 224).to(device)
model = squeezenet1_1(pretrained=False)
model.classifier[1] = Conv2d(model.classifier[1].in_channels, 5, kernel_size=1)
model.to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 5]
