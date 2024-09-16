"""MLP-Mixer, ResMLP, and gMLP in PyTorch

This impl originally based on MLP-Mixer paper.

Official JAX impl: https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py

Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601

@article{tolstikhin2021,
  title={MLP-Mixer: An all-MLP Architecture for Vision},
  author={Tolstikhin, Ilya and Houlsby, Neil and Kolesnikov, Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner,
        Thomas and Yung, Jessica and Keysers, Daniel and Uszkoreit, Jakob and Lucic, Mario and Dosovitskiy, Alexey},
  journal={arXiv preprint arXiv:2105.01601},
  year={2021}
}

Also supporting ResMlp, and a preliminary (not verified) implementations of gMLP

Code: https://github.com/facebookresearch/deit
Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
@misc{touvron2021resmlp,
      title={ResMLP: Feedforward networks for image classification with data-efficient training},
      author={Hugo Touvron and Piotr Bojanowski and Mathilde Caron and Matthieu Cord and Alaaeldin El-Nouby and
        Edouard Grave and Armand Joulin and Gabriel Synnaeve and Jakob Verbeek and Hervé Jégou},
      year={2021},
      eprint={2105.03404},
}

Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
@misc{liu2021pay,
      title={Pay Attention to MLPs},
      author={Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
      year={2021},
      eprint={2105.08050},
}

A thank you to paper authors for releasing code and weights.

Hacked together by / Copyright 2021 Ross Wightman
"""

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import GatedMlp, DropPath, lecun_normal_, to_2tuple, _assert
from timm.layers.format import Format, nchw_to
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq
from timm.models._registry import (
    generate_default_cfgs,
    register_model,
    register_model_deprecations,
)

__all__ = [
    "MixerBlock",
    "MlpMixer",
]  # model_registry will add each entrypoint fn to this


# -------------------------------------#
#   Patch BCHW -> BNC   out=[B, position, channel]
#   [B, 3, 224, 224] -> [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
# -------------------------------------#
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    output_fmt: Format

    def __init__(
        self,
        img_size: int | None = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten: bool = True,
        output_fmt: str | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(
                H == self.img_size[0],
                f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
            )
            _assert(
                W == self.img_size[1],
                f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
            )

        x = self.proj(x)  # [B, 3, 224, 224] -> [B, 768, 14, 14]
        if self.flatten:
            x = x.flatten(2).transpose(
                1, 2
            )  # [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
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

        #                       mix position                            mix channel

    def forward(
        self, x
    ):  # mlp_tokens[B, channel, position]      mlp_channels[B, position, channel]
        x = self.fc1(
            x
        )  # [B, 768, 196] -> [B, 768, 384]        [B, 196, 768] -> [B, 196, 3072]
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(
            x
        )  # [B, 768, 384] -> [B, 768, 196]        [B, 196, 3072] -> [B, 196, 768]
        x = self.drop2(x)
        return x


# -------------------------------------#
#   MLP-Mixer Block
# -------------------------------------#
class MixerBlock(nn.Module):
    """Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """

    def __init__(
        self,
        dim,
        seq_len,
        mlp_ratio=(0.5, 4.0),
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        residual = x
        x = self.norm1(x).transpose(
            1, 2
        )  # [B, 196, 768] -> [B, 768, 196]    mix position
        x = self.mlp_tokens(x).transpose(
            1, 2
        )  # [B, 768, 196] -> [B, 768, 196] -> [B, 196, 768]
        x = residual + self.drop_path(x)

        residual = x
        x = self.mlp_channels(
            self.norm2(x)
        )  # [B, 196, 768] -> [B, 196, 768]    mix channel
        x = residual + self.drop_path(x)
        return x


# -------------------------------------#
#   调整每一个dim
#   beta + alpha * x
# -------------------------------------#
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.ones((1, 1, dim))
        )  # [1, 1, 384] 384是最后的channel
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))  # [1, 1, 384]

    def forward(self, x):
        # 调整每一个channel
        # beta + alpha * x
        # [1, 1, 384] + [1, 1, 384] * [B, 196, 384] = [B, 196, 384] + [B, 196, 384] = [B, 196, 384]
        return torch.addcmul(self.beta, self.alpha, x)


# -------------------------------------#
#   Residual MLP Block
# -------------------------------------#
class ResBlock(nn.Module):
    """Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """

    def __init__(
        self,
        dim,
        seq_len,
        mlp_ratio=4,
        mlp_layer=Mlp,
        norm_layer=Affine,
        act_layer=nn.GELU,
        init_values=1e-4,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)

        # -------------------------------------#
        #   mix position 只使用了Linear
        # -------------------------------------#
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        # 使用了MLP
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)

        # ----------------------------------------------------------------------------------------#
        #   通道权重    形状都为 [384] 与x最后的dim上相乘   三维数据 不需要扩展,直接指定最后维度即可
        #   pytorch两个矩阵数据相乘,假设其中一个矩阵形状为[B, C, H, W],另一个矩阵从后往前有确定形状
        #   指示(通道可以为1)即可相乘比如[1], [W], [H, 1], [C, 1, 1], [1, C, 1, 1],更高维数据也是如此
        # ----------------------------------------------------------------------------------------#
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.norm1(x).transpose(
            1, 2
        )  # [B, 196, 384] -> [B, 384, 196]                            mix position
        x = (
            self.ls1 * self.linear_tokens(x).transpose(1, 2)
        )  # [B, 384, 196] -> [B, 384, 196] -> [B, 196, 384] -> [384] * [B, 196, 384] -> [B, 196, 384]
        x = residual + self.drop_path(x)

        residual = x
        x = self.ls2 * self.mlp_channels(
            self.norm2(x)
        )  # [B, 196, 384] -> [384] * [B, 196, 384] -> [B, 196, 384]   mix channel
        x = residual + self.drop_path(x)
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
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        # 前后2部分哪个部分经过激活函数的区别
        x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
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
#       │       transpose   [B, P, C/2] -> [B, C/2, P]
#       │           │
#       │      proj(seq_len)
#       │           │
#       │       transpose   [B, C/2, P] -> [B, P, C/2]
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
        u, v = x.chunk(2, dim=-1)  # [B, P, C] -> 2 * [B, P, C/2]
        v = self.norm(v)
        v = self.proj(
            v.transpose(-1, -2)
        )  # [B, P, C/2] -> [B, C/2, P] -> [B, C/2, P] 对 seq 维度做投影
        return u * v.transpose(
            -1, -2
        )  # [B, P, C/2] * ([B, C/2, P] -> [B, P, C/2]) = [B, P, C/2]  最终通道减半


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
        x = self.fc1(x)  # [B, 196, 256] -> [B, 196, 1536]
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)  # [B, 196, 1536] -> [B, 196, 768]   add this 通道减半
        x = self.norm(x)  #                                   add this
        x = self.fc2(x)  # [B, 196, 768] -> [B, 196, 256]
        x = self.drop2(x)
        return x


# -------------------------------------#
#   gmlp Block
# -------------------------------------#
class SpatialGatingBlock(nn.Module):
    """Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(
        self,
        dim,
        seq_len,
        mlp_ratio=4,
        mlp_layer=GatedMlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(
            dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class MlpMixer(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        img_size=224,
        in_chans=3,
        patch_size=16,
        num_blocks=8,
        embed_dim=512,
        mlp_ratio=(0.5, 4.0),
        block_layer=MixerBlock,
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        nlhb=False,
        stem_norm=False,
        global_pool="avg",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.grad_checkpointing = False

        # [B, 3, 224, 224] -> [B, 196, 384]
        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
        )

        # 重复 num_blocks 个ResBlock
        # [B, 196, 384] -> [B, 196, 384]
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(
            *[
                block_layer(
                    embed_dim,
                    self.stem.num_patches,
                    mlp_ratio,
                    mlp_layer=mlp_layer,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    drop=proj_drop_rate,
                    drop_path=drop_path_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        )

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.0
        named_apply(
            partial(_init_weights, head_bias=head_bias), module=self
        )  # depth-first

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg")
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        x = self.stem(x)  # [B, 3, 224, 224] -> [B, 196, 384] [B, position, channel]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)  # [B, 196, 384] -> [B, 196, 384]
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == "avg":
            x = x.mean(
                dim=1
            )  # [B, 196, 384] -> [B, 384]    这里在dim=1上求均值,196=14*14,代表了高宽为14的图形,所以相当于求全图的均值
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)  # [B, 384] -> [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)  # [B, 3, 224, 224] -> [B, 196, 384]
        x = self.forward_head(x)  # [B, 196, 384] -> [B, 384]
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0.0, flax=False):
    """Mixer weight initialization (trying to match Flax defaults)"""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if "mlp" in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """Remap checkpoints if needed"""
    if "patch_embed.proj.weight" in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace("patch_embed.", "stem.")
            k = k.replace("attn.", "linear_tokens.")
            k = k.replace("mlp.", "mlp_channels.")
            k = k.replace("gamma_", "ls")
            if k.endswith(".alpha") or k.endswith(".beta"):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict


def _create_mixer(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for MLP-Mixer models.")

    model = build_model_with_cfg(
        MlpMixer,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )
    return model


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "first_conv": "stem.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        "mixer_s32_224.untrained": _cfg(),
        "mixer_s16_224.untrained": _cfg(),
        "mixer_b32_224.untrained": _cfg(),
        "mixer_b16_224.goog_in21k_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth",
        ),
        "mixer_b16_224.goog_in21k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224_in21k-617b3de2.pth",
            num_classes=21843,
        ),
        "mixer_l32_224.untrained": _cfg(),
        "mixer_l16_224.goog_in21k_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth",
        ),
        "mixer_l16_224.goog_in21k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth",
            num_classes=21843,
        ),
        # Mixer ImageNet-21K-P pretraining
        "mixer_b16_224.miil_in21k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil_in21k-2a558a71.pth",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            crop_pct=0.875,
            interpolation="bilinear",
            num_classes=11221,
        ),
        "mixer_b16_224.miil_in21k_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil-9229a591.pth",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            crop_pct=0.875,
            interpolation="bilinear",
        ),
        "gmixer_12_224.untrained": _cfg(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        ),
        "gmixer_24_224.ra3_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmixer_24_224_raa-7daf7ae6.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_12_224.fb_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_24_224.fb_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth",
            # url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resmlp_24_224_raa-a8256759.pth',
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_36_224.fb_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_big_24_224.fb_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_12_224.fb_distilled_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_24_224.fb_distilled_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_36_224.fb_distilled_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_big_24_224.fb_distilled_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_big_24_224.fb_in22k_ft_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_12_224.fb_dino": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlp_12_dino.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "resmlp_24_224.fb_dino": _cfg(
            hf_hub_id="timm/",
            url="https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth",
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
        "gmlp_ti16_224.untrained": _cfg(),
        "gmlp_s16_224.ra3_in1k": _cfg(
            hf_hub_id="timm/",
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth",
        ),
        "gmlp_b16_224.untrained": _cfg(),
    }
)


def mixer_s32_224(pretrained=False, **kwargs) -> MlpMixer:
    """Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer("mixer_s32_224", pretrained=pretrained, **model_args)
    return model


def mixer_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    """Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer("mixer_s16_224", pretrained=pretrained, **model_args)
    return model


def mixer_b32_224(pretrained=False, **kwargs) -> MlpMixer:
    """Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer("mixer_b32_224", pretrained=pretrained, **model_args)
    return model


def mixer_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    """Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer("mixer_b16_224", pretrained=pretrained, **model_args)
    return model


def mixer_l32_224(pretrained=False, **kwargs) -> MlpMixer:
    """Mixer-L/32 224x224.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer("mixer_l32_224", pretrained=pretrained, **model_args)
    return model


def mixer_l16_224(pretrained=False, **kwargs) -> MlpMixer:
    """Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer("mixer_l16_224", pretrained=pretrained, **model_args)
    return model


def gmixer_12_224(pretrained=False, **kwargs) -> MlpMixer:
    """Glu-Mixer-12 224x224
    Experiment by Ross Wightman, adding SwiGLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=16,
        num_blocks=12,
        embed_dim=384,
        mlp_ratio=(1.0, 4.0),
        mlp_layer=GluMlp,
        act_layer=nn.SiLU,
        **kwargs,
    )
    model = _create_mixer("gmixer_12_224", pretrained=pretrained, **model_args)
    return model


def gmixer_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """Glu-Mixer-24 224x224
    Experiment by Ross Wightman, adding SwiGLU to MLP-Mixer
    """
    model_args = dict(
        patch_size=16,
        num_blocks=24,
        embed_dim=384,
        mlp_ratio=(1.0, 4.0),
        mlp_layer=GluMlp,
        act_layer=nn.SiLU,
        **kwargs,
    )
    model = _create_mixer("gmixer_24_224", pretrained=pretrained, **model_args)
    return model


def resmlp_12_224(pretrained=False, **kwargs) -> MlpMixer:
    """ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16,
        num_blocks=12,
        embed_dim=384,
        mlp_ratio=4,
        block_layer=ResBlock,
        norm_layer=Affine,
        **kwargs,
    )
    model = _create_mixer("resmlp_12_224", pretrained=pretrained, **model_args)
    return model


def resmlp_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16,
        num_blocks=24,
        embed_dim=384,
        mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-5),
        norm_layer=Affine,
        **kwargs,
    )
    model = _create_mixer("resmlp_24_224", pretrained=pretrained, **model_args)
    return model


def resmlp_36_224(pretrained=False, **kwargs) -> MlpMixer:
    """ResMLP-36
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=16,
        num_blocks=36,
        embed_dim=384,
        mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6),
        norm_layer=Affine,
        **kwargs,
    )
    model = _create_mixer("resmlp_36_224", pretrained=pretrained, **model_args)
    return model


def resmlp_big_24_224(pretrained=False, **kwargs) -> MlpMixer:
    """ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """
    model_args = dict(
        patch_size=8,
        num_blocks=24,
        embed_dim=768,
        mlp_ratio=4,
        block_layer=partial(ResBlock, init_values=1e-6),
        norm_layer=Affine,
        **kwargs,
    )
    model = _create_mixer("resmlp_big_24_224", pretrained=pretrained, **model_args)
    return model


def gmlp_ti16_224(pretrained=False, **kwargs) -> MlpMixer:
    """gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16,
        num_blocks=30,
        embed_dim=128,
        mlp_ratio=6,
        block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp,
        **kwargs,
    )
    model = _create_mixer("gmlp_ti16_224", pretrained=pretrained, **model_args)
    return model


def gmlp_s16_224(pretrained=False, **kwargs) -> MlpMixer:
    """gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16,
        num_blocks=30,
        embed_dim=256,
        mlp_ratio=6,
        block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp,
        **kwargs,
    )
    model = _create_mixer("gmlp_s16_224", pretrained=pretrained, **model_args)
    return model


def gmlp_b16_224(pretrained=False, **kwargs) -> MlpMixer:
    """gMLP-Base
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model_args = dict(
        patch_size=16,
        num_blocks=30,
        embed_dim=512,
        mlp_ratio=6,
        block_layer=SpatialGatingBlock,
        mlp_layer=GatedMlp,
        **kwargs,
    )
    model = _create_mixer("gmlp_b16_224", pretrained=pretrained, **model_args)
    return model


register_model_deprecations(
    __name__,
    {
        "mixer_b16_224_in21k": "mixer_b16_224.goog_in21k_ft_in1k",
        "mixer_l16_224_in21k": "mixer_l16_224.goog_in21k_ft_in1k",
        "mixer_b16_224_miil": "mixer_b16_224.miil_in21k_ft_in1k",
        "mixer_b16_224_miil_in21k": "mixer_b16_224.miil_in21k",
        "resmlp_12_distilled_224": "resmlp_12_224.fb_distilled_in1k",
        "resmlp_24_distilled_224": "resmlp_24_224.fb_distilled_in1k",
        "resmlp_36_distilled_224": "resmlp_36_224.fb_distilled_in1k",
        "resmlp_big_24_distilled_224": "resmlp_big_24_224.fb_distilled_in1k",
        "resmlp_big_24_224_in22ft1k": "resmlp_big_24_224.fb_in22k_ft_in1k",
        "resmlp_12_224_dino": "resmlp_12_224",
        "resmlp_24_224_dino": "resmlp_24_224",
    },
)


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = gmlp_s16_224(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    # 查看结构
    if False:
        onnx_path = "gmlp_s16_224.onnx"
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
