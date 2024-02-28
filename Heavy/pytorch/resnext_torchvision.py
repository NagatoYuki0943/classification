import torch
from torch import nn
from torchvision import models


device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.ones(1, 3, 224, 224).to(device)
model = models.resnext50_32x4d()
# 修改最后一层的输出
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size()) # [1, 10]
