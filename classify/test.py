import torch
from torch import tensor

a = [
    {"loss": tensor(4.6109, device="cuda:0")},
    {"loss": tensor(4.6039, device="cuda:0")},
    {"loss": tensor(4.6084, device="cuda:0")},
    {"loss": tensor(4.5959, device="cuda:0")},
    {"loss": tensor(4.6067, device="cuda:0")},
]

a = [x["loss"] for x in a]
print(a)


b = [
    tensor(4.6109, device="cuda:0"),
    tensor(4.6039, device="cuda:0"),
    tensor(4.6084, device="cuda:0"),
    tensor(4.5959, device="cuda:0"),
    tensor(4.6067, device="cuda:0"),
]


print(
    torch.stack(b)
)  # tensor([4.6109, 4.6039, 4.6084, 4.5959, 4.6067], device='cuda:0')
print(torch.cat(b))
# Traceback (most recent call last):
#   File "d:/AI/Ai/06_classification/classify/test.py", line 24, in <module>
#     print(torch.cat(b))
# RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated
