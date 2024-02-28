'''
'''

import torch
from torch import nn
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = shufflenet_v2_x1_0()
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]
