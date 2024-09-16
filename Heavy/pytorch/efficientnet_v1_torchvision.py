import torch
from torch import nn
from torchvision import models


device = (
    "cuda:0"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

x = torch.ones(1, 3, 224, 224).to(device)

model = models.efficientnet_b0()
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
model.to(device)

# 有选择的冻住一些层
for name, param in model.named_parameters():
    # 官方自带的使用的是列表,不是字典,所以没有 top 了
    # 除最后全连接层外，其他权重全部冻结
    if "classifier" not in name:
        # param.requires_grad = False   # same as below
        param.requires_grad_(False)
    else:
        print("training {}".format(name))
        # training classifier.1.weight
        # training classifier.1.bias

# 需要优化的参数
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(pg, lr=0.001)

model.eval()
with torch.inference_mode():
    y = model(x)
print(y.size())  # [1, 10]
