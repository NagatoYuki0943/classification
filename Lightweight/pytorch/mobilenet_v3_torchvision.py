import torch
from torch import nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = mobilenet_v3_large()
    # 更改最后的分类层
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)
    # print(model)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]
