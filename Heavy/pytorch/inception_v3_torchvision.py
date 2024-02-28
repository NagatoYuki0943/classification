import torch
from torchvision.models import inception_v3

device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

x = torch.ones(2, 3, 299, 299).to(device)
model = inception_v3(num_classes=10, init_weights=True).to(device)

model.eval()
with torch.inference_mode():
    y, aux  = model(x)
print(y.size())     # [10]
print(aux.size())   # [10]
