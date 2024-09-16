"""EfficientViT (by MSRA)

Paper: `EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention`
    - https://arxiv.org/abs/2305.07027

Adapted from official impl at https://github.com/microsoft/Cream/tree/main/EfficientViT

只有3个stage,开始下采样16倍,然后有2次下采样,一共下采样64倍
"""

__all__ = ["EfficientVitMsra"]
import itertools
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import (
    SqueezeExcite,
    SelectAdaptivePool2d,
    trunc_normal_,
    _assert,
)
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs


# --------------------------------------------#
#   Overlap PatchEmbed
#   kernel = 3
#   stride = 2,2,2,2
#          = 16 下采样16倍,类似 vit_patch16
# --------------------------------------------#
class PatchEmbedding(nn.Sequential):
    def __init__(self, in_chans, dim):
        super().__init__()
        self.add_module("conv1", ConvNorm(in_chans, dim // 8, 3, 2, 1))
        self.add_module("relu1", nn.ReLU())
        self.add_module("conv2", ConvNorm(dim // 8, dim // 4, 3, 2, 1))
        self.add_module("relu2", nn.ReLU())
        self.add_module("conv3", ConvNorm(dim // 4, dim // 2, 3, 2, 1))
        self.add_module("relu3", nn.ReLU())
        self.add_module("conv4", ConvNorm(dim // 2, dim, 3, 2, 1))
        self.patch_size = 16


class ConvNorm(nn.Sequential):
    def __init__(
        self,
        in_chs,
        out_chs,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, ks, stride, pad, dilation, groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chs)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(
            w.size(1) * self.conv.groups,
            w.size(0),
            w.shape[2:],
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# -----------#
#   分类头
# -----------#
class NormLinear(nn.Sequential):
    def __init__(self, in_features, out_features, bias=True, std=0.02, drop=0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(drop)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        trunc_normal_(self.linear.weight, std=std)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, linear = self.bn, self.linear
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = (
            bn.bias
            - self.bn.running_mean * self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        )
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = b @ self.linear.weight.T
        else:
            b = (linear.weight @ b[:, None]).view(-1) + self.linear.bias
        m = nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


# -----------------------------------------------#
#   下采样层
#   通过一个类似MobileNetV3的Block实现的,没有残差
# -----------------------------------------------#
class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = ConvNorm(dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = ConvNorm(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, 0.25)
        self.conv3 = ConvNorm(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


# ----------------------------------------------------#
#   随机丢弃batch数据,类似DropPath,不过添加了Residual
# ----------------------------------------------------#
class ResidualDrop(nn.Module):
    def __init__(self, m, drop=0.0):
        super().__init__()
        self.m = m  # ConvNorm
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return (
                x
                + self.m(x)
                * torch.rand(x.size(0), 1, 1, 1, device=x.device)
                .ge_(self.drop)
                .div(1 - self.drop)
                .detach()
            )
        else:
            return x + self.m(x)


# ----------------------------------------#
#   Feed Forward Network
#   1x1ConvNorm -> ReLU -> 1x1ConvNorm
# ----------------------------------------#
class ConvMlp(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = ConvNorm(ed, h)
        self.act = nn.ReLU()
        self.pw2 = ConvNorm(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


# ------------------------------------------------------#
#   级联Attn,每个级联相当于一个head
#             in
#              │
#              │
#    ┌────── split ──────┐ split 4 branches as heads
#    │     ┌──┘ └──┐     │
#   attn   │       │     │
#    │     │       │     │
#    ├──── +       │     │
#    │     │       │     │
#    │    attn     │     │
#    │     │       │     │
#    │     ├────── +     │
#    │     │       │     │
#    │     │      attn   │
#    │     │       │     │
#    │     │       ├──── +
#    │     │       │     │
#    │     └──┐ ┌──┘    attn
#    │        │ │        │
#    └────── concat ─────┘
#              │
#             proj
#              │
#             out
#
#       attn
#                        in
#                         │
#                        qkv
#                         │
#               ┌────── split ──────────────────┐
#               │         └─────────────┐       │
#               │                       │       │
#               q                       k       v(v的通道数可能比q,k大,q和k的通道永远相同)
#               │                       │       │
#  Token Interaction(DWConvNorm)        │       │
#               │                       │       │
#               │         ┌─────────────┘       │
#               │         │                     │
#               └──── self-attn ────────────────┘
#                         │
#                        out
# ------------------------------------------------------#
class CascadedGroupAttention(nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,  # 三个stage都为4 必须和kernels数量相等
        attn_ratio=4,
        resolution=14,
        kernels=(5, 5, 5, 5),  # 几个kernel就有几个级联
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            # ConvNorm
            qkvs.append(ConvNorm(dim // (num_heads), self.key_dim * 2 + self.val_dim))
            # DWConvNorm
            dws.append(
                ConvNorm(
                    self.key_dim,
                    self.key_dim,
                    kernels[i],
                    1,
                    kernels[i] // 2,
                    groups=self.key_dim,
                )
            )
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(
            nn.ReLU(), ConvNorm(self.val_dim * num_heads, dim, bn_weight_init=0)
        )

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )
        self.register_buffer(
            "attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False
        )
        self.attention_bias_cache = {}

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[
                    :, self.attention_bias_idxs
                ]
            return self.attention_bias_cache[device_key]

    def forward(self, x):
        B, C, H, W = x.shape
        feats_in = x.chunk(
            len(self.qkvs), dim=1
        )  # [B, C, H, W] -> n * [B, c, H, W]  C = n * c
        feats_out = []
        feat = feats_in[0]  # 开始的feat,后面的feat = 前面的输出 + 当前的输入
        attn_bias = self.get_attention_biases(x.device)  # [n, H*W, H*W]
        for head_idx, (qkv, dws) in enumerate(zip(self.qkvs, self.dws)):
            if head_idx > 0:
                feat = feat + feats_in[head_idx]  # feat = 前面的输出 + 当前的输入
            feat = qkv(
                feat
            )  # [B, c, H, W] -> [B, 3*c, H, W] 不一定是3*c,q和k一定相等,v不一定
            q, k, v = feat.view(B, -1, H, W).split(
                [self.key_dim, self.key_dim, self.val_dim], dim=1
            )  # [B, 3*c, H, W] -> 3 * [B, c, H, W]
            q = dws(q)  # [B, c, H, W] -> [B, c, H, W]  Token Interaction(DWConvNorm)
            q, k, v = (
                q.flatten(2),
                k.flatten(2),
                v.flatten(2),
            )  # [B, c, H, W] -> [B, c, H*W]
            q = q * self.scale
            attn = (
                q.transpose(-2, -1) @ k
            )  # ([B, c, H*W] -> [B, H*W, c]) @ [B, c, H*W] = [B, H*W, H*W]
            attn = (
                attn + attn_bias[head_idx]
            )  # [B, H*W, H*W] + [1, H*W, H*W] = [B, H*W, H*W]
            attn = attn.softmax(dim=-1)
            feat = v @ attn.transpose(
                -2, -1
            )  # [B, c, H*W] @ ([B, H*W, H*W] -> [B, H*W, H*W]) = [B, c, H*W]
            feat = feat.view(B, self.val_dim, H, W)  # [B, c, H*W] -> [B, c, H, W]
            feats_out.append(feat)
        x = self.proj(
            torch.cat(feats_out, 1)
        )  # n * [B, c, H, W] cat [B, C, H, W] -> [B, C, H, W]
        return x


# --------------------------------------------------#
#   划分窗口,使用CascadedGroupAttention,然后还原窗口
# --------------------------------------------------#
class LocalWindowAttention(nn.Module):
    r"""Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        window_resolution=7,
        kernels=(5, 5, 5, 5),
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, "window_size must be greater than 0"
        self.window_resolution = window_resolution
        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(
            dim,
            key_dim,
            num_heads,
            attn_ratio=attn_ratio,
            resolution=window_resolution,
            kernels=kernels,
        )

    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        _assert(
            H == H_, f"input feature has wrong size, expect {(H, W)}, got {(H_, W_)}"
        )
        _assert(
            W == W_, f"input feature has wrong size, expect {(H, W)}, got {(H_, W_)}"
        )

        # H = W = 7 or 4 直接进入这里,4不会被padding
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
            pad_b = (
                self.window_resolution - H % self.window_resolution
            ) % self.window_resolution
            pad_r = (
                self.window_resolution - W % self.window_resolution
            ) % self.window_resolution
            x = nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution

            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            # [B, pH, pW, C] -> [B, nH, h, nW, w, C] -> [B, nH, nW, h, w, C]
            x = x.view(
                B, nH, self.window_resolution, nW, self.window_resolution, C
            ).transpose(2, 3)
            # [B, nH, nW, h, w, C] -> [B*nH*nW, h, w, C] -> [B*nH*nW, C, h, w]
            x = x.reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)

            # CascadedGroupAttention
            # [B*nH*nW, C, h, w] -> [B*nH*nW, C, h, w]
            x = self.attn(x)

            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            # [B*nH*nW, C, h, w] -> [B*nH*nW, h, w, C] -> [B, nH, nW, h, w, C]
            x = x.permute(0, 2, 3, 1).view(
                B, nH, nW, self.window_resolution, self.window_resolution, C
            )
            # [B, nH, nW, h, w, C] -> [B, nH, h, nW, w, C] ->[B, pH, pW, C]
            x = x.transpose(2, 3).reshape(B, pH, pW, C)
            # [B, pH, pW, C] 去除padding [B, H, W, C]
            x = x[:, :H, :W].contiguous()
            # [B, H, W, C] -> [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        return x


# ----------------------------------------#
#   EfficientVitBlock Stage中多次调用
#                    in
#                     │
#      ┌──────────────┤
#      │   Token Interaction(DWConvNorm)
#      │              │
#      └───────────── +
#                     │
#      ┌──────────────┤
#      │      Feed Forward Network
#      │              │
#      └───────────── +
#                     │
#      ┌──────────────┤
#      │   Cascaded Group Attention
#      │              │
#      └───────────── +
#                     │
#      ┌──────────────┤
#      │   Token Interaction(DWConvNorm)
#      │              │
#      └───────────── +
#                     │
#      ┌──────────────┤
#      │      Feed Forward Network
#      │              │
#      └───────────── +
#                     │
#                    out
# ----------------------------------------#
class EfficientVitBlock(nn.Module):
    """A basic EfficientVit building block.

    Args:
        dim (int): Number of input channels.
        key_dim (int): Dimension for query and key in the token mixer.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        window_resolution=7,
        kernels=[5, 5, 5, 5],
    ):
        super().__init__()
        # DWConvNorm
        self.dw0 = ResidualDrop(
            ConvNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.0)
        )
        self.ffn0 = ResidualDrop(ConvMlp(dim, int(dim * 2)))

        self.mixer = ResidualDrop(
            LocalWindowAttention(
                dim,
                key_dim,
                num_heads,
                attn_ratio=attn_ratio,
                resolution=resolution,
                window_resolution=window_resolution,
                kernels=kernels,
            )
        )
        # DWConvNorm
        self.dw1 = ResidualDrop(
            ConvNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.0)
        )
        self.ffn1 = ResidualDrop(ConvMlp(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


# -----------------------------#
#   3个stage,每个stage使用1次
# -----------------------------#
class EfficientVitStage(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        key_dim,
        downsample=("", 1),
        num_heads=8,
        attn_ratio=4,
        resolution=14,
        window_resolution=7,
        kernels=[5, 5, 5, 5],
        depth=1,
    ):
        super().__init__()
        if downsample[0] == "subsample":
            self.resolution = (resolution - 1) // downsample[1] + 1
            down_blocks = []
            down_blocks.append(
                (
                    "res1",
                    nn.Sequential(
                        ResidualDrop(ConvNorm(in_dim, in_dim, 3, 1, 1, groups=in_dim)),
                        ResidualDrop(ConvMlp(in_dim, int(in_dim * 2))),
                    ),
                )
            )
            down_blocks.append(("patchmerge", PatchMerging(in_dim, out_dim)))
            down_blocks.append(
                (
                    "res2",
                    nn.Sequential(
                        ResidualDrop(
                            ConvNorm(out_dim, out_dim, 3, 1, 1, groups=out_dim)
                        ),
                        ResidualDrop(ConvMlp(out_dim, int(out_dim * 2))),
                    ),
                )
            )
            self.downsample = nn.Sequential(OrderedDict(down_blocks))
        else:
            assert in_dim == out_dim
            self.downsample = nn.Identity()
            self.resolution = resolution

        blocks = []
        for d in range(depth):
            blocks.append(
                EfficientVitBlock(
                    out_dim,
                    key_dim,
                    num_heads,
                    attn_ratio,
                    self.resolution,
                    window_resolution,
                    kernels,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):  #           第一次不用下采样
        x = self.downsample(
            x
        )  # [B, 64, 14, 14] -> [B, 64, 14, 14] -> [B, 128, 7, 7] -> [B, 192, 4, 4]
        x = self.blocks(x)
        return x


class EfficientVitMsra(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dim=(64, 128, 192),
        key_dim=(16, 16, 16),
        depth=(1, 2, 3),
        num_heads=(4, 4, 4),
        window_size=(7, 7, 7),
        kernels=(5, 5, 5, 5),
        down_ops=(("", 1), ("subsample", 2), ("subsample", 2)),
        global_pool="avg",
        drop_rate=0.0,
    ):
        super(EfficientVitMsra, self).__init__()
        self.grad_checkpointing = False
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # ------------------------#
        #   Overlap PatchEmbed
        #   下采样16倍
        # ------------------------#
        self.patch_embed = PatchEmbedding(in_chans, embed_dim[0])
        stride = self.patch_embed.patch_size
        resolution = img_size // self.patch_embed.patch_size
        attn_ratio = [
            embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))
        ]

        # Build EfficientVit blocks
        self.feature_info = []
        stages = []
        pre_ed = embed_dim[0]
        for i, (ed, kd, dpth, nh, ar, wd, do) in enumerate(
            zip(embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)
        ):
            stage = EfficientVitStage(
                in_dim=pre_ed,
                out_dim=ed,
                key_dim=kd,
                downsample=do,
                num_heads=nh,
                attn_ratio=ar,
                resolution=resolution,
                window_resolution=wd,
                kernels=kernels,
                depth=dpth,
            )
            pre_ed = ed
            if do[0] == "subsample" and i != 0:
                stride *= do[1]
            resolution = stage.resolution
            stages.append(stage)
            self.feature_info += [
                dict(num_chs=ed, reduction=stride, module=f"stages.{i}")
            ]
        self.stages = nn.Sequential(*stages)

        if global_pool == "avg":
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        else:
            assert num_classes == 0
            self.global_pool = nn.Identity()
        self.num_features = embed_dim[-1]
        self.head = (
            NormLinear(self.num_features, num_classes, drop=self.drop_rate)
            if num_classes > 0
            else nn.Identity()
        )

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            if global_pool == "avg":
                self.global_pool = SelectAdaptivePool2d(
                    pool_type=global_pool, flatten=True
                )
            else:
                assert num_classes == 0
                self.global_pool = nn.Identity()
        self.head = (
            NormLinear(self.num_features, num_classes, drop=self.drop_rate)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x):
        x = self.patch_embed(
            x
        )  # [B, 3, 224, 224] -> [B, 64, 14, 14] 直接下采样16倍,类似 vit_patch16
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)  # [B, 64, 14, 14] -> [B, 192, 4, 4]
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)  # [B, 192, 4, 4] -> [B, 192]
        return x if pre_logits else self.head(x)  # [B, 192] -> [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)  # [B, 3, 224, 224] -> [B, 192, 4, 4]
        x = self.forward_head(x)  # [B, 192, 4, 4] -> [B, num_classes]
        return x


# def checkpoint_filter_fn(state_dict, model):
#     if 'model' in state_dict.keys():
#         state_dict = state_dict['model']
#     tmp_dict = {}
#     out_dict = {}
#     target_keys = model.state_dict().keys()
#     target_keys = [k for k in target_keys if k.startswith('stages.')]
#
#     for k, v in state_dict.items():
#         if 'attention_bias_idxs' in k:
#             continue
#         k = k.split('.')
#         if k[-2] == 'c':
#             k[-2] = 'conv'
#         if k[-2] == 'l':
#             k[-2] = 'linear'
#         k = '.'.join(k)
#         tmp_dict[k] = v
#
#     for k, v in tmp_dict.items():
#         if k.startswith('patch_embed'):
#             k = k.split('.')
#             k[1] = 'conv' + str(int(k[1]) // 2 + 1)
#             k = '.'.join(k)
#         elif k.startswith('blocks'):
#             kw = '.'.join(k.split('.')[2:])
#             find_kw = [a for a in list(sorted(tmp_dict.keys())) if kw in a]
#             idx = find_kw.index(k)
#             k = [a for a in target_keys if kw in a][idx]
#         out_dict[k] = v
#
#     return out_dict


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.conv1.conv",
        "classifier": "head.linear",
        "fixed_input_size": True,
        "pool_size": (4, 4),
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        "efficientvit_m0.r224_in1k": _cfg(
            hf_hub_id="timm/",
            # url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m0.pth'
        ),
        "efficientvit_m1.r224_in1k": _cfg(
            hf_hub_id="timm/",
            # url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m1.pth'
        ),
        "efficientvit_m2.r224_in1k": _cfg(
            hf_hub_id="timm/",
            # url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m2.pth'
        ),
        "efficientvit_m3.r224_in1k": _cfg(
            hf_hub_id="timm/",
            # url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m3.pth'
        ),
        "efficientvit_m4.r224_in1k": _cfg(
            hf_hub_id="timm/",
            # url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m4.pth'
        ),
        "efficientvit_m5.r224_in1k": _cfg(
            hf_hub_id="timm/",
            # url='https://github.com/xinyuliu-jeffrey/EfficientVit_Model_Zoo/releases/download/v1.0/efficientvit_m5.pth'
        ),
    }
)


def _create_efficientvit_msra(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop("out_indices", (0, 1, 2))
    model = build_model_with_cfg(
        EfficientVitMsra,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs,
    )
    return model


def efficientvit_m0(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[64, 128, 192],  # 3 stage
        depth=[1, 2, 3],
        num_heads=[4, 4, 4],
        window_size=[7, 7, 7],
        kernels=[5, 5, 5, 5],
    )
    return _create_efficientvit_msra(
        "efficientvit_m0", pretrained=pretrained, **dict(model_args, **kwargs)
    )


def efficientvit_m1(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[128, 144, 192],  # 3 stage
        depth=[1, 2, 3],
        num_heads=[2, 3, 3],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3],
    )
    return _create_efficientvit_msra(
        "efficientvit_m1", pretrained=pretrained, **dict(model_args, **kwargs)
    )


def efficientvit_m2(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[128, 192, 224],  # 3 stage
        depth=[1, 2, 3],
        num_heads=[4, 3, 2],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3],
    )
    return _create_efficientvit_msra(
        "efficientvit_m2", pretrained=pretrained, **dict(model_args, **kwargs)
    )


def efficientvit_m3(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[128, 240, 320],  # 3 stage
        depth=[1, 2, 3],
        num_heads=[4, 3, 4],
        window_size=[7, 7, 7],
        kernels=[5, 5, 5, 5],
    )
    return _create_efficientvit_msra(
        "efficientvit_m3", pretrained=pretrained, **dict(model_args, **kwargs)
    )


def efficientvit_m4(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[128, 256, 384],  # 3 stage
        depth=[1, 2, 3],
        num_heads=[4, 4, 4],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3],
    )
    return _create_efficientvit_msra(
        "efficientvit_m4", pretrained=pretrained, **dict(model_args, **kwargs)
    )


def efficientvit_m5(pretrained=False, **kwargs):
    model_args = dict(
        img_size=224,
        embed_dim=[192, 288, 384],  # 3 stage
        depth=[1, 3, 4],
        num_heads=[3, 3, 4],
        window_size=[7, 7, 7],
        kernels=[7, 5, 3, 3],
    )
    return _create_efficientvit_msra(
        "efficientvit_m5", pretrained=pretrained, **dict(model_args, **kwargs)
    )


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = efficientvit_m0(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    if False:
        onnx_path = "efficientvit_m0.onnx"
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
