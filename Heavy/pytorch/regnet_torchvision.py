import torch
from torch import nn
from torchvision import models


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = models.regnet_y_400mf()
    # 调整输出维度
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 10]
