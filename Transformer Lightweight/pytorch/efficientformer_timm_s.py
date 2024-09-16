"""EfficientFormer

@article{li2022efficientformer,
  title={EfficientFormer: Vision Transformers at MobileNet Speed},
  author={Li, Yanyu and Yuan, Geng and Wen, Yang and Hu, Eric and Evangelidis, Georgios and Tulyakov,
   Sergey and Wang, Yanzhi and Ren, Jian},
  journal={arXiv preprint arXiv:2206.01191},
  year={2022}
}

Based on Apache 2.0 licensed code at https://github.com/snap-research/EfficientFormer, Copyright (c) 2022 Snap Inc.

Modifications and timm support by / Copyright 2022, Ross Wightman
"""

from typing import Dict
from functools import partial
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, to_2tuple
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import generate_default_cfgs, register_model

__all__ = ["EfficientFormer"]  # model_registry will add each entrypoint fn to this


EfficientFormer_width = {
    "l1": (48, 96, 224, 448),
    "l3": (64, 128, 320, 512),
    "l7": (96, 192, 384, 768),
}

EfficientFormer_depth = {
    "l1": (3, 2, 6, 4),
    "l3": (4, 4, 12, 6),
    "l7": (6, 6, 18, 8),
}


# -----------------------------------------------------------------------#
#   stage1,2,3只会使用 pool + conv_mlp
#   stage4会使用 pool + conv_mlp 和 self_attn + mlp
# -----------------------------------------------------------------------#


# ---------------------------------------------------------------#
#   开始的stem
#   [B, 3, 224, 224] -> [B, 24, 112, 112] -> [B, 48, 56, 56]
# ---------------------------------------------------------------#
class Stem4(nn.Sequential):
    def __init__(self, in_chs, out_chs, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.stride = 4

        self.add_module(
            "conv1", nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1)
        )
        self.add_module("norm1", norm_layer(out_chs // 2))
        self.add_module("act1", act_layer())
        self.add_module(
            "conv2",
            nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        )
        self.add_module("norm2", norm_layer(out_chs))
        self.add_module("act2", act_layer())


# --------------------#
#   stage的下采样层
# --------------------#
class Downsample(nn.Module):
    """
    Downsampling via strided conv w/ norm
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=3,
        stride=2,
        padding=None,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(out_chs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


# -----------------------#
#   MetaBlock2d 中使用
#   stride为1
# -----------------------#
class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        return self.pool(x) - x


# -----------------------------------------------------#
#   [B, C, H, W] -> [B, n*C, H, W] -> [B, C, H, W]
# -----------------------------------------------------#
class ConvMlpWithNorm(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.norm1 = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.norm2 = (
            norm_layer(out_features) if norm_layer is not None else nn.Identity()
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # [B, C, H, W] -> [B, 4*C, H, W]
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # [B, 4*C, H, W] -> [B, C, H, W]
        x = self.norm2(x)
        x = self.drop(x)
        return x


# -------------------------------------------------#
#   通道倍率
#   [B, C, H, W] * [1, C, 1, 1] = [B, C, H, W]
# -------------------------------------------------#
class LayerScale2d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)  # [C] -> [1, C, 1, 1]
        return (
            x.mul_(gamma) if self.inplace else x * gamma
        )  # [B, C, H, W] * [1, C, 1, 1] = [B, C, H, W]


# ---------------------------------------------#
#   pool and conv_mlp
#   [B, C, H, W]
#   前3个stage和stage4的前几个block使用
#
#       x
#       │
#      pool
#       │
#    conv_mlp
#       │
# ---------------------------------------------#
class MetaBlock2d(nn.Module):
    def __init__(
        self,
        dim,
        pool_size=3,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.token_mixer = Pooling(pool_size=pool_size)
        self.ls1 = LayerScale2d(dim, layer_scale_init_value)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = ConvMlpWithNorm(
            dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale2d(dim, layer_scale_init_value)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(
            self.ls1(self.token_mixer(x))
        )  # [B, C, H, W] -> [B, C, H, W]  pooling(k=3)
        x = x + self.drop_path2(
            self.ls2(self.mlp(x))
        )  # [B, C, H, W] -> [B, 4*C, H, W] -> [B, C, H, W]
        return x


# ------------------------------------------------#
#   [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
# ------------------------------------------------#
class Flat(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(torch.nn.Module):
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(self, dim=384, key_dim=32, num_heads=8, attn_ratio=4, resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio

        self.qkv = nn.Linear(dim, self.key_attn_dim * 2 + self.val_attn_dim)
        self.proj = nn.Linear(self.val_attn_dim, dim)

        resolution = to_2tuple(resolution)
        pos = torch.stack(
            torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))
        ).flatten(1)
        rel_pos = (pos[..., :, None] - pos[..., None, :]).abs()
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, resolution[0] * resolution[1])
        )
        self.register_buffer("attention_bias_idxs", torch.LongTensor(rel_pos))
        self.attention_bias_cache = {}  # per-device attention_biases cache (data-parallel compat)

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

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, 49, 448] -> [B, 49, 1536]
        qkv = qkv.reshape(B, N, self.num_heads, -1)  # [B, 49, 1536] -> [B, 49, 8, 192]
        qkv = qkv.permute(0, 2, 1, 3)  # [B, 49, 8, 192] -> [B, 8, 49, 192]
        q, k, v = qkv.split(
            [self.key_dim, self.key_dim, self.val_dim], dim=3
        )  # [B, 8, 49, 192] -> [B, 8, 49, 32], [B, 8, 49, 32], [B, 8, 49, 128]

        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale  # [B, 8, 49, 32] @ [B, 8, 32, 49] = [B, 8, 49, 49]
        attn = attn + self.get_attention_biases(
            x.device
        )  # [B, 8, 49, 49] + [8, 49, 49] = [B, 8, 49, 49]
        attn = attn.softmax(dim=-1)  # 取每一列,在行上做softmax

        x = attn @ v  # [B, 8, 49, 49] @ [B, 8, 49, 128] = [B, 8, 49, 128]
        x = x.transpose(1, 2)  # [B, 8, 49, 128] -> [B, 49, 8, 128]
        x = x.reshape(B, N, self.val_attn_dim)  # [B, 49, 8, 128] -> [B, 49, 1024]

        x = self.proj(x)  # [B, 49, 1024] -> [B, 49, 448]
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


# ---------------------------------------#
#   通道倍率
#   [B, N, C] * [1, 1, C] = [B, N, C]
# ---------------------------------------#
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# ---------------------------------------------#
#   self attn and mlp
#   [B, N, C]
#   stage4后面使用
#
#       x
#       │
#     linear
#       │
#   ┌───┼───┐
#   │   │   │
#   V  K^T  Q
#   │   │   │
#   └──attn─┘  attn使用了多余的get_attention_biases
#       │
#     linear
#       │
#   linear_mlp
#       │
# ---------------------------------------------#
class MetaBlock1d(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = LayerScale(dim, layer_scale_init_value)
        self.ls2 = LayerScale(dim, layer_scale_init_value)

    def forward(self, x):
        x = x + self.drop_path(
            self.ls1(self.token_mixer(self.norm1(x)))
        )  # [B, N, C] -> [B, N, C]
        x = x + self.drop_path(
            self.ls2(self.mlp(self.norm2(x)))
        )  # [B, N, C] -> [B, N, 4*C] -> [B, N, C]
        return x


# --------------------------#
#   4个stage中的每个stage
# --------------------------#
class EfficientFormerStage(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        depth,
        downsample=True,
        num_vit=1,  # 使用几次 MetaBlock1d vit 模块
        pool_size=3,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        norm_layer_cl=nn.LayerNorm,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if downsample:
            self.downsample = Downsample(
                in_chs=dim, out_chs=dim_out, norm_layer=norm_layer
            )
            dim = dim_out
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []

        # 如果全为vit,就提前展平数据
        if num_vit and num_vit >= depth:
            blocks.append(Flat())  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]

        for block_idx in range(depth):
            remain_idx = depth - block_idx - 1  # id从0开始,因此要多减一
            if num_vit and num_vit > remain_idx:  # stage最后才使用vit
                blocks.append(
                    MetaBlock1d(  # [B, N, C] stage4
                        dim,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer_cl,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            else:
                blocks.append(
                    MetaBlock2d(  # [B, C, H, W] stage1 stage2 stage3
                        dim,
                        pool_size=pool_size,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                if num_vit and num_vit == remain_idx:  # 使用vit之前展平数据
                    blocks.append(Flat())  # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):  # 第一次没有下采样
        x = self.downsample(
            x
        )  # [B, 48, 56, 56] -> [B, 48, 56, 56] -> [B, 96, 28, 28] -> [B, 224, 14, 14] -> [B, 448, 7, 7]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)  # same
        return x


class EfficientFormer(nn.Module):
    def __init__(
        self,
        depths,  # [3, 2, 6, 4]       stage depths
        embed_dims=None,  # [48, 96, 224, 448] stage dims
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        downsamples=None,
        num_vit=0,  # 使用几次 MetaBlock1d vit 模块
        mlp_ratios=4,  # mlp ratio
        pool_size=3,  # MetaBlock2d pool size
        layer_scale_init_value=1e-5,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm2d,
        norm_layer_cl=nn.LayerNorm,
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        # [B, 3, 224, 224] -> [B, 24, 112, 112] -> [B, 48, 56, 56]
        self.stem = Stem4(in_chans, embed_dims[0], norm_layer=norm_layer)
        prev_dim = embed_dims[0]

        # stochastic depth decay rule
        dpr = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        downsamples = downsamples or (False,) + (True,) * (len(depths) - 1)
        stages = []
        for i in range(len(depths)):
            stage = EfficientFormerStage(
                prev_dim,
                embed_dims[i],
                depths[i],
                downsample=downsamples[i],
                num_vit=num_vit
                if i == 3
                else 0,  # 最后的stage使用attn, 参数是使用 attn block 的次数
                pool_size=pool_size,
                mlp_ratio=mlp_ratios,
                act_layer=act_layer,
                norm_layer_cl=norm_layer_cl,
                norm_layer=norm_layer,
                proj_drop=proj_drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
            )
            prev_dim = embed_dims[i]
            stages.append(stage)

        self.stages = nn.Sequential(*stages)

        # Classifier head
        self.num_features = embed_dims[-1]
        self.norm = norm_layer_cl(self.num_features)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        # assuming model is always distilled (valid for current checkpoints, will split def if that changes)
        self.head_dist = (
            nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.distilled_training = (
            False  # must set this True to train w/ distillation token
        )

        self.apply(self._init_weights)

    # init for classification
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x):
        x = self.stem(x)  # [B, 3, 224, 224] -> [B, 24, 112, 112] -> [B, 48, 56, 56]
        x = self.stages(
            x
        )  # [B, 48, 56, 56] -> [B, 96, 28, 28] -> [B, 224, 14, 14] -> [B, 49, 448]
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == "avg":
            x = x.mean(dim=1)  # [B, 49, 448] -> [B, 448]
        x = self.head_drop(x)
        if pre_logits:
            return x
        x, x_dist = (
            self.head(x),
            self.head_dist(x),
        )  # [B, 448] -> [B, num_classes] and [B, 448] -> [B, num_classes]
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train/finetune, inference average the classifier predictions
            return (
                x + x_dist
            ) / 2  # ([B, num_classes] + [B, num_classes]) / 2 = [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)  # [B, 3, 224, 224] -> [B, 49, 448]
        x = self.forward_head(x)  # [B, 49, 448] -> [B, num_classes]
        return x


def _checkpoint_filter_fn(state_dict, model):
    """Remap original checkpoints -> timm"""
    if "stem.0.weight" in state_dict:
        return state_dict  # non-original checkpoint, no remapping needed

    out_dict = {}
    import re

    stage_idx = 0
    for k, v in state_dict.items():
        if k.startswith("patch_embed"):
            k = k.replace("patch_embed.0", "stem.conv1")
            k = k.replace("patch_embed.1", "stem.norm1")
            k = k.replace("patch_embed.3", "stem.conv2")
            k = k.replace("patch_embed.4", "stem.norm2")

        if re.match(r"network\.(\d+)\.proj\.weight", k):
            stage_idx += 1
        k = re.sub(r"network.(\d+).(\d+)", f"stages.{stage_idx}.blocks.\\2", k)
        k = re.sub(r"network.(\d+).proj", f"stages.{stage_idx}.downsample.conv", k)
        k = re.sub(r"network.(\d+).norm", f"stages.{stage_idx}.downsample.norm", k)

        k = re.sub(r"layer_scale_([0-9])", r"ls\1.gamma", k)
        k = k.replace("dist_head", "head_dist")
        out_dict[k] = v
    return out_dict


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "fixed_input_size": True,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "stem.conv1",
        "classifier": ("head", "head_dist"),
        **kwargs,
    }


default_cfgs = generate_default_cfgs(
    {
        "efficientformer_l1.snap_dist_in1k": _cfg(
            hf_hub_id="timm/",
        ),
        "efficientformer_l3.snap_dist_in1k": _cfg(
            hf_hub_id="timm/",
        ),
        "efficientformer_l7.snap_dist_in1k": _cfg(
            hf_hub_id="timm/",
        ),
    }
)


def _create_efficientformer(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        EfficientFormer,
        variant,
        pretrained,
        pretrained_filter_fn=_checkpoint_filter_fn,
        **kwargs,
    )
    return model


def efficientformer_l1(pretrained=False, **kwargs) -> EfficientFormer:
    model_args = dict(
        depths=EfficientFormer_depth["l1"],
        embed_dims=EfficientFormer_width["l1"],
        num_vit=1,  # stage4 最后1个block使用vit,前3次使用卷积
    )
    return _create_efficientformer(
        "efficientformer_l1", pretrained=pretrained, **dict(model_args, **kwargs)
    )


def efficientformer_l3(pretrained=False, **kwargs) -> EfficientFormer:
    model_args = dict(
        depths=EfficientFormer_depth["l3"],
        embed_dims=EfficientFormer_width["l3"],
        num_vit=4,  # stage4 最后4个block使用vit,前2次使用卷积
    )
    return _create_efficientformer(
        "efficientformer_l3", pretrained=pretrained, **dict(model_args, **kwargs)
    )


def efficientformer_l7(pretrained=False, **kwargs) -> EfficientFormer:
    model_args = dict(
        depths=EfficientFormer_depth["l7"],
        embed_dims=EfficientFormer_width["l7"],
        num_vit=8,  # stage4 最后8个block使用vit,也就是全部使用vit
    )
    return _create_efficientformer(
        "efficientformer_l7", pretrained=pretrained, **dict(model_args, **kwargs)
    )


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = efficientformer_l1(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    if False:
        onnx_path = "efficientformer_l1.onnx"
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
