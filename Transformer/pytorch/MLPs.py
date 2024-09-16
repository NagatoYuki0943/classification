"""MLP module w/ dropout and configurable activation layer

https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple
from timm.layers import Mlp, GluMlp, SwiGLU, GatedMlp, ConvMlp


# -----------------------------------------#
#   [..., C] -> [..., n*C] -> [..., C]
# -----------------------------------------#
class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):  # mix channel
        x = self.fc1(x)  # [B, N, C] -> [B, N, n*C]
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)  # [B, N, n*C] -> [B, N, C]
        x = self.drop2(x)
        return x


# -----------------------#
#          in
#           │
#           │
#          fc1
#           │
#     ┌───split───┐
#    act          │
#     └──── * ────┘
#           │
#         drop1
#           │
#         norm
#           │
#          fc2
#           │
#         drop2
#           │
#           │
#          out
# -----------------------#
class GluMlp(nn.Module):
    """MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Sigmoid,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features // 2)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # [B, N, C] -> [B, N, n*C]
        x1, x2 = x.chunk(2, dim=self.chunk_dim)  # [B, N, n*C] -> 2 * [B, N, n*C/2]
        # 前后2部分哪个部分经过激活函数的区别
        x = (
            x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        )  # [B, N, n*C/2] * act([B, N, n*C/2]) = [B, N, n*C/2]
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)  # [B, N, n*C/2] -> [B, N, C]
        x = self.drop2(x)
        return x


# -----------------------#
#          in
#           │
#           │
#     ┌─────┴─────┐
#     │           │
#   fc1_g         │
#     │         fc1_x
#    act          │
#     │           │
#     └──── * ────┘
#           │
#         drop1
#           │
#         norm
#           │
#          fc2
#           │
#         drop2
#           │
#           │
#          out
# -----------------------#
class SwiGLU(nn.Module):
    """SwiGLU
    NOTE: GluMLP above can implement SwiGLU, but this impl has split fc1 and
    better matches some other common impl which makes mapping checkpoints simpler.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)  # [B, N, C] -> [B, N, n*C]
        x = self.fc1_x(x)  # [B, N, C] -> [B, N, n*C]
        x = self.act(x_gate) * x  # act([B, N, n*C]) * [B, N, n*C] = [B, N, n*C]
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)  # [B, N, n*C] -> [B, N, C]
        x = self.drop2(x)
        return x


# -------------------------------#
#   GatedMlp使用的gate_layer
#            in
#             │
#             │
#       ┌── split ──┐
#       │           │
#       │         norm
#       │           │
#       │       transpose   [B, N, C/2] -> [B, C/2, P]
#       │           │
#       │      proj(seq_len)
#       │           │
#       │       transpose   [B, C/2, P] -> [B, N, C/2]
#       │           │
#       └──── * ────┘
#             │
#             │
#            out
# -------------------------------#
class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)  # [B, N, C] -> 2 * [B, N, C/2]
        v = self.norm(v)
        v = self.proj(
            v.transpose(-1, -2)
        )  # [B, N, C/2] -> [B, C/2, P] -> [B, C/2, P] 对 seq 维度做投影
        return u * v.transpose(
            -1, -2
        )  # [B, N, C/2] * ([B, C/2, P] -> [B, N, C/2]) = [B, N, C/2]  最终通道减半


# -----------------------#
#          in
#           │
#           │
#          fc1
#           │
#          act
#           │
#         drop1
#           │
#         gate: SpatialGatingUnit
#           │
#         norm
#           │
#          fc2
#           │
#         drop2
#           │
#           │
#          out
# -----------------------#
class GatedMlp(nn.Module):
    """MLP as used in gMLP"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        gate_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = (
                hidden_features // 2
            )  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)  # [B, N, C] -> [B, N, n*C]
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)  # [B, N, n*C] -> [B, N, n*C/2]   add this 通道减半
        x = self.norm(x)
        x = self.fc2(x)  # [B, N, n*C/2] -> [B, N, C]
        x = self.drop2(x)
        return x


# -----------------------#
#   1x1Conv代替全连接层
# -----------------------#
class ConvMlp(nn.Module):
    """MLP using 1x1 convs that keeps spatial dims"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        # -------------------------------------#
        #   使用k=1的Conv代替两个全连接层
        # -------------------------------------#
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)  # [B, C, H, W] -> [B, n*C, H, W]
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # [B, n*C, H, W] -> [B, C, H, W]
        return x


if __name__ == "__main__":
    x = torch.ones(1, 256, 64, 64)
    x_ = x.flatten(2).transpose(1, 2)  # [1, 4096, 256]

    mlp = Mlp(in_features=256, hidden_features=256 * 4, out_features=256).eval()
    glumlp = GluMlp(in_features=256, hidden_features=256 * 4, out_features=256).eval()
    swiglu = SwiGLU(in_features=256, hidden_features=256 * 4, out_features=256).eval()
    gate_layer = partial(SpatialGatingUnit, seq_len=x_.shape[1])
    gatedmlp = GatedMlp(
        in_features=256,
        hidden_features=256 * 4,
        out_features=256,
        gate_layer=gate_layer,
    ).eval()
    convmlp = ConvMlp(in_features=256, hidden_features=256 * 4, out_features=256).eval()

    with torch.inference_mode():
        print(mlp(x_).shape)  # [1, 4096, 256]
        print(glumlp(x_).shape)  # [1, 4096, 256]
        print(swiglu(x_).shape)  # [1, 4096, 256]
        print(gatedmlp(x_).shape)  # [1, 4096, 256]
        print(convmlp(x).shape)  # [1, 256, 64, 64]
