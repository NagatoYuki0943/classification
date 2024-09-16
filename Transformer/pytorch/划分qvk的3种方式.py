"""划分qvk的3种方式"""

import torch
from torch import nn

POSITION = 196
CHANNEL = 768
HEAD = 12

fc = nn.Linear(CHANNEL, CHANNEL * 3)

x = torch.ones(1, POSITION, CHANNEL)
x = fc(x)  # [1, 196, 768] -> [1, 196, 2304]


# ------------------------------------#
#   permute(2, 0, 3, 1, 4) + ubind
#   unbind 和 [0] [1] [2] 等价
# ------------------------------------#
y1 = x.reshape(
    1, POSITION, 3, HEAD, CHANNEL // HEAD
)  # [1, 196, 2304] -> [1, 196, 3, 12, 64]
y1 = y1.permute(2, 0, 3, 1, 4)  # [1, 196, 3, 12, 64] -> [3, 1, 12, 196, 64]
q1, k1, v1 = y1.unbind(0)  # [3, 1, 12, 196, 64] -> [1, 12, 196, 64] * 3
print(q1.shape, k1.shape, v1.shape)


# ------------------------------------#
#   不把3放到最前面,直接unbind
# ------------------------------------#
y2 = x.reshape(
    1, POSITION, 3, HEAD, CHANNEL // HEAD
)  # [1, 196, 2304] -> [1, 196, 3, 12, 64]
y2 = y2.transpose(1, 3)  # [1, 196, 3, 12, 64] -> [1, 12, 3, 196, 64]
q2, k2, v2 = y2.unbind(2)  # [1, 12, 3, 196, 64] -> [1, 12, 196, 64] * 3
print(q2.shape, k2.shape, v2.shape)


# ------------------------------------#
#   不分出3来,直接chunk
# ------------------------------------#
y3 = x.reshape(
    1, POSITION, HEAD, CHANNEL * 3 // HEAD
)  # [1, 196, 2304] -> [1, 196, 12, 192]
y3 = y3.transpose(1, 2)  # [1, 196, 12, 196] -> [1, 12, 196, 192]
q3, k3, v3 = y3.chunk(chunks=3, dim=3)  # [1, 12, 196, 192] -> [1, 12, 196, 64] * 3
print(q3.shape, k3.shape, v3.shape)
