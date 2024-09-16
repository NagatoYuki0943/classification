"""MobileViT

Paper:
V1: `MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer` - https://arxiv.org/abs/2110.02178
V2: `Separable Self-attention for Mobile Vision Transformers` - https://arxiv.org/abs/2206.02680

MobileVitBlock and checkpoints adapted from https://github.com/apple/ml-cvnets (original copyright below)
License: https://github.com/apple/ml-cvnets/blob/main/LICENSE (Apple open source)

Rest of code, ByobNet, and Transformer block hacked together by / Copyright 2022, Ross Wightman
"""

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import math
from functools import partial
from typing import Callable, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.jit import Final

from timm.layers import (
    to_2tuple,
    make_divisible,
    GroupNorm1,
    DropPath,
    is_exportable,
    use_fused_attn,
)
from timm.layers.fast_norm import is_fast_norm, fast_group_norm
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_module
from timm.models._registry import (
    register_model,
    generate_default_cfgs,
    register_model_deprecations,
)
from timm.models.byobnet import (
    register_block,
    ByoBlockCfg,
    ByoModelCfg,
    ByobNet,
    LayerFn,
    num_groups,
)

__all__ = []


def _inverted_residual_block(d, c, s, br=4.0):
    # inverted residual is a bottleneck block with bottle_ratio > 1 applied to in_chs, linear output, gs=1 (depthwise)
    return ByoBlockCfg(
        type="bottle",
        d=d,
        c=c,
        s=s,
        gs=1,
        br=br,
        block_kwargs=dict(bottle_in=True, linear_out=True),
    )


def _mobilevit_block(d, c, s, transformer_dim, transformer_depth, patch_size=4, br=4.0):
    # inverted residual + mobilevit blocks as per MobileViT network
    return (
        _inverted_residual_block(d=d, c=c, s=s, br=br),
        ByoBlockCfg(
            type="mobilevit",
            d=1,
            c=c,
            s=1,
            block_kwargs=dict(
                transformer_dim=transformer_dim,
                transformer_depth=transformer_depth,
                patch_size=patch_size,
            ),
        ),
    )


def _mobilevitv2_block(
    d, c, s, transformer_depth, patch_size=2, br=2.0, transformer_br=0.5
):
    # inverted residual + mobilevit blocks as per MobileViT network
    return (
        _inverted_residual_block(d=d, c=c, s=s, br=br),
        ByoBlockCfg(
            type="mobilevit2",
            d=1,
            c=c,
            s=1,
            br=transformer_br,
            gs=1,
            block_kwargs=dict(
                transformer_depth=transformer_depth, patch_size=patch_size
            ),
        ),
    )


def _mobilevitv2_cfg(multiplier=1.0):
    chs = (64, 128, 256, 384, 512)
    if multiplier != 1.0:
        chs = tuple([int(c * multiplier) for c in chs])
    cfg = ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=chs[0], s=1, br=2.0),
            _inverted_residual_block(d=2, c=chs[1], s=2, br=2.0),
            _mobilevitv2_block(d=1, c=chs[2], s=2, transformer_depth=2),
            _mobilevitv2_block(d=1, c=chs[3], s=2, transformer_depth=4),
            _mobilevitv2_block(d=1, c=chs[4], s=2, transformer_depth=3),
        ),
        stem_chs=int(32 * multiplier),
        stem_type="3x3",
        stem_pool="",
        downsample="",
        act_layer="silu",
    )
    return cfg


model_cfgs = dict(
    mobilevit_xxs=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=16, s=1, br=2.0),
            _inverted_residual_block(d=3, c=24, s=2, br=2.0),
            _mobilevit_block(
                d=1,
                c=48,
                s=2,
                transformer_dim=64,
                transformer_depth=2,
                patch_size=2,
                br=2.0,
            ),
            _mobilevit_block(
                d=1,
                c=64,
                s=2,
                transformer_dim=80,
                transformer_depth=4,
                patch_size=2,
                br=2.0,
            ),
            _mobilevit_block(
                d=1,
                c=80,
                s=2,
                transformer_dim=96,
                transformer_depth=3,
                patch_size=2,
                br=2.0,
            ),
        ),
        stem_chs=16,
        stem_type="3x3",
        stem_pool="",
        downsample="",
        act_layer="silu",
        num_features=320,
    ),
    mobilevit_xs=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=48, s=2),
            _mobilevit_block(
                d=1, c=64, s=2, transformer_dim=96, transformer_depth=2, patch_size=2
            ),
            _mobilevit_block(
                d=1, c=80, s=2, transformer_dim=120, transformer_depth=4, patch_size=2
            ),
            _mobilevit_block(
                d=1, c=96, s=2, transformer_dim=144, transformer_depth=3, patch_size=2
            ),
        ),
        stem_chs=16,
        stem_type="3x3",
        stem_pool="",
        downsample="",
        act_layer="silu",
        num_features=384,
    ),
    mobilevit_s=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=64, s=2),
            _mobilevit_block(
                d=1, c=96, s=2, transformer_dim=144, transformer_depth=2, patch_size=2
            ),
            _mobilevit_block(
                d=1, c=128, s=2, transformer_dim=192, transformer_depth=4, patch_size=2
            ),
            _mobilevit_block(
                d=1, c=160, s=2, transformer_dim=240, transformer_depth=3, patch_size=2
            ),
        ),
        stem_chs=16,
        stem_type="3x3",
        stem_pool="",
        downsample="",
        act_layer="silu",
        num_features=640,
    ),
    semobilevit_s=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=64, s=2),
            _mobilevit_block(
                d=1, c=96, s=2, transformer_dim=144, transformer_depth=2, patch_size=2
            ),
            _mobilevit_block(
                d=1, c=128, s=2, transformer_dim=192, transformer_depth=4, patch_size=2
            ),
            _mobilevit_block(
                d=1, c=160, s=2, transformer_dim=240, transformer_depth=3, patch_size=2
            ),
        ),
        stem_chs=16,
        stem_type="3x3",
        stem_pool="",
        downsample="",
        attn_layer="se",
        attn_kwargs=dict(rd_ratio=1 / 8),
        num_features=640,
    ),
    mobilevitv2_050=_mobilevitv2_cfg(0.50),
    mobilevitv2_075=_mobilevitv2_cfg(0.75),
    mobilevitv2_125=_mobilevitv2_cfg(1.25),
    mobilevitv2_100=_mobilevitv2_cfg(1.0),
    mobilevitv2_150=_mobilevitv2_cfg(1.5),
    mobilevitv2_175=_mobilevitv2_cfg(1.75),
    mobilevitv2_200=_mobilevitv2_cfg(2.0),
)


# -------------------------#
#   vit中的attn
#   MobileVitV1使用
# -------------------------#
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, C] -> [B, N, 3*C]
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, self.head_dim
        )  # [B, N, 3*C] -> [B, N, 3, h, c]    C = h * c
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [B, N, 3, h, c] -> [3, B, h, N, c]
        q, k, v = qkv.unbind(0)  # [3, B, h, N, c] -> 3 * [B, h, N, c]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(  # f(q,k,v) = [B, h, N, c]
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, h, N, c] @ [B, h, c, N] = [B, h, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, h, N, N] @ [B, h, N, c] = [B, h, N, c]

        x = x.transpose(1, 2)  # [B, h, N, c] -> [B, N, h, c]
        x = x.reshape(B, N, C)  # [B, N, h, c] -> [B, N, C]

        x = self.proj(x)  # [B, N, C] -> [B, N, C]
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# -------------------------------------#
#   1x1Conv代替全连接层
#   宽高不为1,不是注意力
# -------------------------------------#
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


# -------------------------#
#   vit中的Block
#   MobileVitV1使用
# -------------------------#
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# -------------------------------------#
#   分组数为1的GN就是LN
#   GN vs LN example:
#       x = torch.ones(1, 3, 224, 224)
#       gn = nn.GroupNorm(1, 3)                         # 分为1组(等价LN),通道为3,数据是4维的
#       print(gn(x).size())                             # [1, 3, 224, 224]
#       ln = nn.LayerNorm([3, 224, 224])                # LN对于4维数据在最后3维上处理,要把 CHW 都写进参数, 对于NLP的3维,会在最后的dim维度上处理
#       print(ln(x).size())                             # [1, 3, 224, 224]
#
#       # 实际使用LN处理图片一般会把图片的形状转换为mlp的形状 [batch, position, channel],将channel调至最后,在channel上计算LN,计算完再转换回来形状
#       ln = nn.LayerNorm(3)
#       x = x.reshape(1, 3, -1).transpose(1, 2)         # [1, 3, 224, 224] -> [1, 3, 224*224] -> [1, 224*224, 3]
#       y = ln(x)
#       y = y.transpose(1, 2).reshape(1, 3, 224, 224)   # [1, 224*224, 3] -> [1, 3, 224*224] -> [1, 3, 224, 224]
#
#       # mlp序列处理实例
#       x = torch.ones(1, 196, 768)
#       ln = nn.LayerNorm(768)                          # 处理最后的dim维度
#       print(ln(x).size())                             # [1, 196, 768]
# -------------------------------------#
class GroupNorm1(nn.GroupNorm):
    """Group Normalization with 1 group.
    Input: tensor in shape [B, C, *]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
        self.fast_norm = (
            is_fast_norm()
        )  # can't script unless we have these flags here (no globals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fast_norm:
            return fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        else:
            return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)


# ---------------------------#
#   MobileVitV1 stage
# ---------------------------#
@register_notrace_module
class MobileVitBlock(nn.Module):
    """MobileViT block
    Paper: https://arxiv.org/abs/2110.02178?context=cs.LG
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        bottle_ratio: float = 1.0,
        group_size: Optional[int] = None,
        dilation: Tuple[int, int] = (1, 1),
        mlp_ratio: float = 2.0,  # mlp_ratio = 2.0
        transformer_dim: Optional[int] = None,
        transformer_depth: int = 2,
        patch_size: int = 8,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        drop: int = 0.0,
        no_fusion: bool = False,
        drop_path_rate: float = 0.0,
        layers: LayerFn = None,
        transformer_norm_layer: Callable = nn.LayerNorm,
        **kwargs,  # eat unused args
    ):
        super(MobileVitBlock, self).__init__()

        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)

        self.conv_kxk = layers.conv_norm_act(
            in_chs,
            in_chs,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            dilation=dilation[0],
        )
        self.conv_1x1 = nn.Conv2d(in_chs, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(
            *[
                TransformerBlock(
                    transformer_dim,
                    mlp_ratio=mlp_ratio,
                    num_heads=num_heads,
                    qkv_bias=True,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    drop_path=drop_path_rate,
                    act_layer=layers.act,
                    norm_layer=transformer_norm_layer,
                )
                for _ in range(transformer_depth)
            ]
        )
        self.norm = transformer_norm_layer(transformer_dim)

        self.conv_proj = layers.conv_norm_act(
            transformer_dim, out_chs, kernel_size=1, stride=1
        )

        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = layers.conv_norm_act(
                in_chs + out_chs, out_chs, kernel_size=kernel_size, stride=1
            )

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.conv_kxk(x)  # [B, 96, 32, 32] -> [B, 96, 32, 32]
        x = self.conv_1x1(x)  # [B, 96, 32, 32] -> [B, 144, 32, 32] 增多通道

        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size  # [2, 2]
        B, C, H, W = x.shape
        # new_h = H / 2, new_w = W / 2
        new_h, new_w = (
            math.ceil(H / patch_h) * patch_h,
            math.ceil(W / patch_w) * patch_w,
        )

        # 大小不同重采样
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False
        if new_h != H or new_w != W:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            interpolate = True

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(
            B * C * num_patch_h, patch_h, num_patch_w, patch_w
        )  # [B, 144, 32, 32] -> [B*144*16, 2, 16, 2]
        x = x.transpose(1, 2)  # [B*144*16, 2, 16, 2] -> [B*144*16, 16, 2, 2]

        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(
            B, C, num_patches, self.patch_area
        )  # [B*144*16, 16, 2, 2] -> [B, 144, 16*16, 2*2]
        x = x.transpose(1, 3)  # [B, 144, 16*16, 2*2] -> [B, 2*2, 16*16, 144]
        x = x.reshape(
            B * self.patch_area, num_patches, -1
        )  # [B, 2*2, 16*16, 144] -> [B*2*2, 16*16, 144]

        # Global representations
        x = self.transformer(x)  # [B*2*2, 16*16, 144] -> [B*2*2, 16*16, 144]
        x = self.norm(x)

        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(
            B, self.patch_area, num_patches, -1
        )  # [B*2*2, 16*16, 144] -> [B, 2*2, 16*16, 144]
        x = x.transpose(1, 3)  # [B, 2*2, 16*16, 144] -> [B, 144, 16*16, 2*2]
        x = x.reshape(
            B * C * num_patch_h, num_patch_w, patch_h, patch_w
        )  # [B, 144, 16*16, 2*2] -> [B*144*16, 16, 2, 2]

        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        x = x.transpose(1, 2)  # [B*144*16, 16, 2, 2] -> [B*144*16, 2, 16, 2]
        x = x.reshape(
            B, C, num_patch_h * patch_h, num_patch_w * patch_w
        )  # [B*144*16, 2, 16, 2] -> [B, 144, 32, 32]

        # 如果使用了采样要还原回去
        if interpolate:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.conv_proj(x)  # [B, 144, 32, 32] -> [B, 96, 32, 32] 减少通道
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x


# -----------------------------------#
#   MobileVitV2使用
#   LinearTransformerBlock用的attn
# -----------------------------------#
class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `https://arxiv.org/abs/2206.02680`
    This layer can be used for self- as well as cross-attention.
    Args:
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_drop (float): Dropout value for context scores. Default: 0.0
        bias (bool): Use bias in learnable layers. Default: True
    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input
    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
        )
        self.out_drop = nn.Dropout(proj_drop)

    def _forward_self_attn(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)  # [B, 128, 4, 256] -> [B, 257, 4, 256]

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = qkv.split(
            [1, self.embed_dim, self.embed_dim], dim=1
        )  # [B, 257, 4, 256] -> [B, 1, 4, 256], [B, 128, 4, 256], [B, 128, 4, 256]

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)  # [B, 1, 4, 256]
        context_scores = self.attn_drop(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(
            dim=-1, keepdim=True
        )  # [B, 128, 4, 256] * [B, 1, 4, 256] = [B, 128, 4, 256] -> [B, 128, 4, 1]

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(
            value
        )  # [B, 128, 4, 256] * [B, 128, 4, 1] = [B, 128, 4, 256]
        out = self.out_proj(out)  # [B, 128, 4, 256] -> [B, 128, 4, 256]
        out = self.out_drop(out)
        return out

    @torch.jit.ignore()
    def _forward_cross_attn(
        self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x --> [B, C, P, N]
        # x_prev = [B, C, P, M]
        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape
        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        # compute query, key, and value
        # [B, C, P, M] --> [B, 1 + d, P, M]
        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.weight[: self.embed_dim + 1],
            bias=self.qkv_proj.bias[: self.embed_dim + 1],
        )

        # [B, 1 + d, P, M] --> [B, 1, P, M], [B, d, P, M]
        query, key = qk.split([1, self.embed_dim], dim=1)
        # [B, C, P, N] --> [B, d, P, N]
        value = F.conv2d(
            x,
            weight=self.qkv_proj.weight[self.embed_dim + 1],
            bias=self.qkv_proj.bias[self.embed_dim + 1]
            if self.qkv_proj.bias is not None
            else None,
        )

        # apply softmax along M dimension
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_drop(context_scores)

        # compute context vector
        # [B, d, P, M] * [B, 1, P, M] -> [B, d, P, M] --> [B, d, P, 1]
        context_vector = (key * context_scores).sum(dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        out = self.out_drop(out)
        return out

    def forward(
        self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x_prev is None:
            return self._forward_self_attn(x)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev)


# -------------------------------------#
#   MobileVitV2Block中的transformer
# -------------------------------------#
class LinearTransformerBlock(nn.Module):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 paper <>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        mlp_ratio (float): Inner dimension ratio of the FFN relative to embed_dim
        drop (float): Dropout rate. Default: 0.0
        attn_drop (float): Dropout rate for attention in multi-head attention. Default: 0.0
        drop_path (float): Stochastic depth rate Default: 0.0
        norm_layer (Callable): Normalization layer. Default: layer_norm_2d
    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        act_layer = act_layer or nn.SiLU
        norm_layer = norm_layer or GroupNorm1

        self.norm1 = norm_layer(embed_dim)
        self.attn = LinearSelfAttention(
            embed_dim=embed_dim, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = norm_layer(embed_dim)
        self.mlp = ConvMlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self, x: torch.Tensor, x_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.drop_path1(
                self.attn(self.norm1(x))
            )  # [B, 128, 4, 256] -> [B, 128, 4, 256]
        else:
            # cross-attention
            res = x
            x = self.norm1(x)  # norm
            x = self.attn(x, x_prev)  # attn
            x = self.drop_path1(x) + res  # residual

        # Feed forward network
        x = x + self.drop_path2(
            self.mlp(self.norm2(x))
        )  # [B, 128, 4, 256] -> [B, 2*128, 4, 256] -> [B, 128, 4, 256]
        return x


# -------------------------#
#   MobileVitV2 stage
# -------------------------#
@register_notrace_module
class MobileVitV2Block(nn.Module):
    """
    This class defines the `MobileViTv2 block <>`_
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: int = 3,
        bottle_ratio: float = 1.0,
        group_size: Optional[int] = 1,
        dilation: Tuple[int, int] = (1, 1),
        mlp_ratio: float = 2.0,
        transformer_dim: Optional[int] = None,
        transformer_depth: int = 2,
        patch_size: int = 8,
        attn_drop: float = 0.0,
        drop: int = 0.0,
        drop_path_rate: float = 0.0,
        layers: LayerFn = None,
        transformer_norm_layer: Callable = GroupNorm1,
        **kwargs,  # eat unused args
    ):
        super(MobileVitV2Block, self).__init__()
        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)

        self.conv_kxk = layers.conv_norm_act(
            in_chs,
            in_chs,
            kernel_size=kernel_size,
            stride=1,
            groups=groups,
            dilation=dilation[0],
        )
        self.conv_1x1 = nn.Conv2d(in_chs, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(
            *[
                LinearTransformerBlock(
                    transformer_dim,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path_rate,
                    act_layer=layers.act,
                    norm_layer=transformer_norm_layer,
                )
                for _ in range(transformer_depth)
            ]
        )
        self.norm = transformer_norm_layer(transformer_dim)

        self.conv_proj = layers.conv_norm_act(
            transformer_dim, out_chs, kernel_size=1, stride=1, apply_act=False
        )

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
        self.coreml_exportable = is_exportable()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape  # [B, 256, 32, 32]
        patch_h, patch_w = self.patch_size  # [2, 2]
        new_h, new_w = (
            math.ceil(H / patch_h) * patch_h,
            math.ceil(W / patch_w) * patch_w,
        )
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N

        # 大小不同重采样
        if new_h != H or new_w != W:
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )

        # Local representation
        x = self.conv_kxk(x)  # [B, 256, 32, 32] -> [B, 256, 32, 32]
        x = self.conv_1x1(x)  # [B, 256, 32, 32] -> [B, 128, 32, 32]

        # Unfold (feature map -> patches), [B, C, H, W] -> [B, C, P, N]
        C = x.shape[1]
        if self.coreml_exportable:
            x = F.unfold(x, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        else:
            x = x.reshape(
                B, C, num_patch_h, patch_h, num_patch_w, patch_w
            )  # [B, 128, 32, 32] -> [B, 128, 16, 2, 16, 2]
            x = x.permute(
                0, 1, 3, 5, 2, 4
            )  # [B, 128, 16, 2, 16, 2] -> [B, 128, 2, 2, 16, 16]
        x = x.reshape(
            B, C, -1, num_patches
        )  # [B, 128, 2, 2, 16, 16] -> [B, 128, 2*2, 16*16]

        # Global representations
        x = self.transformer(x)  # [B, 128, 2*2, 16*16] -> [B, 128, 2*2, 16*16]
        x = self.norm(x)

        # Fold (patches -> feature map), [B, C, P, N] --> [B, C, H, W]
        if self.coreml_exportable:
            # adopted from https://github.com/apple/ml-cvnets/blob/main/cvnets/modules/mobilevit_block.py#L609-L624
            x = x.reshape(B, C * patch_h * patch_w, num_patch_h, num_patch_w)
            x = F.pixel_shuffle(x, upscale_factor=patch_h)
        else:
            x = x.reshape(
                B, C, patch_h, patch_w, num_patch_h, num_patch_w
            )  # [B, 128, 2*2, 16*16] -> [B, 128, 2, 2, 16, 16]
            x = x.permute(
                0, 1, 4, 2, 5, 3
            )  # [B, 128, 2, 2, 16, 16] -> [B, 128, 16, 2, 16, 2]
            x = x.reshape(
                B, C, num_patch_h * patch_h, num_patch_w * patch_w
            )  # [B, 128, 16, 2, 16, 2] -> [B, 128, 32, 32]

        x = self.conv_proj(x)  # [B, 128, 32, 32] -> [B, 256, 32, 32]
        return x


#################
#   for debug   #
#################
from dataclasses import replace
from timm.models.byobnet import (
    get_layer_fns,
    create_byob_stem,
    reduce_feat_size,
    create_byob_stages,
    _init_weights,
)
from timm.models._manipulate import named_apply, checkpoint_seq
from timm.layers.classifier import ClassifierHead


class ByobNet(nn.Module):
    """'Bring-your-own-blocks' Net

    A flexible network backbone that allows building model stem + blocks via
    dataclass cfg definition w/ factory functions for module instantiation.

    Current assumption is that both stem and blocks are in conv-bn-act order (w/ block ending in act).
    """

    def __init__(
        self,
        cfg: ByoModelCfg,
        num_classes: int = 1000,
        in_chans: int = 3,
        global_pool: str = "avg",
        output_stride: int = 32,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        zero_init_last: bool = True,
        **kwargs,
    ):
        """
        Args:
            cfg: Model architecture configuration.
            num_classes: Number of classifier classes.
            in_chans: Number of input channels.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            img_size: Image size for fixed image size models (i.e. self-attn).
            drop_rate: Classifier dropout rate.
            drop_path_rate: Stochastic depth drop-path rate.
            zero_init_last: Zero-init last weight of residual path.
            **kwargs: Extra kwargs overlayed onto cfg.
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        cfg = replace(cfg, **kwargs)  # overlay kwargs onto cfg
        layers = get_layer_fns(cfg)
        if cfg.fixed_input_size:
            assert (
                img_size is not None
            ), "img_size argument is required for fixed input size model"
        feat_size = to_2tuple(img_size) if img_size is not None else None

        self.feature_info = []
        stem_chs = int(round((cfg.stem_chs or cfg.blocks[0].c) * cfg.width_factor))
        self.stem, stem_feat = create_byob_stem(
            in_chans, stem_chs, cfg.stem_type, cfg.stem_pool, layers=layers
        )
        self.feature_info.extend(stem_feat[:-1])
        feat_size = reduce_feat_size(feat_size, stride=stem_feat[-1]["reduction"])

        self.stages, stage_feat = create_byob_stages(
            cfg,
            drop_path_rate,
            output_stride,
            stem_feat[-1],
            layers=layers,
            feat_size=feat_size,
        )
        self.feature_info.extend(stage_feat[:-1])

        prev_chs = stage_feat[-1]["num_chs"]
        if cfg.num_features:
            self.num_features = int(round(cfg.width_factor * cfg.num_features))
            self.final_conv = layers.conv_norm_act(prev_chs, self.num_features, 1)
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.feature_info += [
            dict(
                num_chs=self.num_features,
                reduction=stage_feat[-1]["reduction"],
                module="final_conv",
            )
        ]

        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
        )

        # init weights
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)  # [B, 3, 256, 256] -> [B, 16, 128, 128]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)  # [B, 16, 128, 128] -> [B, 160, 8, 8]
        x = self.final_conv(x)  # [B, 160, 8, 8] -> [B, 640, 8, 8]
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)  # [B, 3, 256, 256] -> [B, 640, 8, 8]
        x = self.forward_head(x)  # [B, 640, 8, 8] -> [B, num_classes]
        return x


register_block("mobilevit", MobileVitBlock)
register_block("mobilevit2", MobileVitV2Block)


def _create_mobilevit(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet,
        variant,
        pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


def _create_mobilevit2(variant, cfg_variant=None, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ByobNet,
        variant,
        pretrained,
        model_cfg=model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant],
        feature_cfg=dict(flatten_sequential=True),
        **kwargs,
    )


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": (8, 8),
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "first_conv": "stem.conv",
        "classifier": "head.fc",
        "fixed_input_size": False,
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        "mobilevit_xxs.cvnets_in1k": _cfg(hf_hub_id="timm/"),
        "mobilevit_xs.cvnets_in1k": _cfg(hf_hub_id="timm/"),
        "mobilevit_s.cvnets_in1k": _cfg(hf_hub_id="timm/"),
        "mobilevitv2_050.cvnets_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_075.cvnets_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_100.cvnets_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_125.cvnets_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_150.cvnets_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_175.cvnets_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_200.cvnets_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_150.cvnets_in22k_ft_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_175.cvnets_in22k_ft_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_200.cvnets_in22k_ft_in1k": _cfg(hf_hub_id="timm/", crop_pct=0.888),
        "mobilevitv2_150.cvnets_in22k_ft_in1k_384": _cfg(
            hf_hub_id="timm/",
            input_size=(3, 384, 384),
            pool_size=(12, 12),
            crop_pct=1.0,
        ),
        "mobilevitv2_175.cvnets_in22k_ft_in1k_384": _cfg(
            hf_hub_id="timm/",
            input_size=(3, 384, 384),
            pool_size=(12, 12),
            crop_pct=1.0,
        ),
        "mobilevitv2_200.cvnets_in22k_ft_in1k_384": _cfg(
            hf_hub_id="timm/",
            input_size=(3, 384, 384),
            pool_size=(12, 12),
            crop_pct=1.0,
        ),
    }
)


def mobilevit_xxs(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevit_xxs", pretrained=pretrained, **kwargs)


def mobilevit_xs(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevit_xs", pretrained=pretrained, **kwargs)


def mobilevit_s(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevit_s", pretrained=pretrained, **kwargs)


def mobilevitv2_050(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevitv2_050", pretrained=pretrained, **kwargs)


def mobilevitv2_075(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevitv2_075", pretrained=pretrained, **kwargs)


def mobilevitv2_100(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevitv2_100", pretrained=pretrained, **kwargs)


def mobilevitv2_125(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevitv2_125", pretrained=pretrained, **kwargs)


def mobilevitv2_150(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevitv2_150", pretrained=pretrained, **kwargs)


def mobilevitv2_175(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevitv2_175", pretrained=pretrained, **kwargs)


def mobilevitv2_200(pretrained=False, **kwargs) -> ByobNet:
    return _create_mobilevit("mobilevitv2_200", pretrained=pretrained, **kwargs)


register_model_deprecations(
    __name__,
    {
        "mobilevitv2_150_in22ft1k": "mobilevitv2_150.cvnets_in22k_ft_in1k",
        "mobilevitv2_175_in22ft1k": "mobilevitv2_175.cvnets_in22k_ft_in1k",
        "mobilevitv2_200_in22ft1k": "mobilevitv2_200.cvnets_in22k_ft_in1k",
        "mobilevitv2_150_384_in22ft1k": "mobilevitv2_150.cvnets_in22k_ft_in1k_384",
        "mobilevitv2_175_384_in22ft1k": "mobilevitv2_175.cvnets_in22k_ft_in1k_384",
        "mobilevitv2_200_384_in22ft1k": "mobilevitv2_200.cvnets_in22k_ft_in1k_384",
    },
)


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 256, 256).to(device)
    model = mobilevit_s(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    # 查看结构
    if False:
        onnx_path = "mobilevit_s.onnx"
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=["images"],
            output_names=["classes"],
        )
        import onnx
        from onnxsim import simplify

        # 载入onnx模型
        model_ = onnx.load(onnx_path)

        # 简化模型
        model_simple, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simple, onnx_path)
        print("finished exporting " + onnx_path)
