'''
densenet自带的预训练模型没法使用torch.load()加载,不过自己训练的模型可以加载
121 161 169 201
'''

import torch
from torch import nn
from torchvision import models


device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.ones(1, 3, 224, 224).to(device)
# densenet自带的预训练模型没法使用torch.load()加载,不过自己训练的模型可以加载
# pre_weights = torch.load(r"D:\AI\预训练权重\densenet169-b2777c0a.pth")
model = models.densenet121()
# model.load_state_dict(pre_weights)

model.classifier = nn.Linear(model.classifier.in_features, 10)
model.to(device)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())
