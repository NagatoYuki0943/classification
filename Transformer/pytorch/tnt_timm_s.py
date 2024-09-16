"""Transformer in Transformer (TNT) in PyTorch

A PyTorch implement of TNT as described in
'Transformer in Transformer' - https://arxiv.org/abs/2103.00112

The official mindspore code is released and available at
https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/TNT
"""

import math
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_, _assert, to_2tuple
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from timm.models.vision_transformer import resize_pos_embed

__all__ = ["TNT"]  # model_registry will add each entrypoint fn to this


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "pixel_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "tnt_s_patch16_224": _cfg(
        url="https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "tnt_b_patch16_224": _cfg(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


class PixelEmbed(nn.Module):
    """Image to Pixel Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_dim=48, stride=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # grid_size property necessary for resizing positional embedding
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = (self.grid_size[0]) * (self.grid_size[1])
        self.img_size = img_size
        self.num_patches = num_patches
        self.in_dim = in_dim
        self.stride = stride
        new_patch_size = [math.ceil(ps / stride) for ps in patch_size]
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2d(
            in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride
        )
        self.unfold = nn.Unfold(kernel_size=new_patch_size, stride=new_patch_size)

    def forward(self, x, pixel_pos):
        """
        Args:
            x (Tensor): Image   ex: [B, 3, 224, 224]
            pixel_pos (Tensor):   ex: [1, 24,  4,  4]
        Returns:
            Tensor: [B*position, 每个position的小patch数量, 每个小patch的channel] ex [B*196, 16, 24]
        """
        B, C, H, W = x.shape  # [B, 3, 224, 224]
        _assert(
            H == self.img_size[0],
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).",
        )
        _assert(
            W == self.img_size[1],
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).",
        )
        x = self.proj(x)  # [B, 3, 224, 224] -> [B, 24, 56, 56]

        # -----------------------------------------------------------------------------------------------------------------------#
        # 尝试自己写unfold 宽高为56,每个由长度为24的channel表示,然后将宽高每4个划分为一个,所以宽高为56/4=14,channel变为24*4*4=384
        # [56] -> [14, 4], 14代表划分为14个,每个的长度为4
        new_patch_size = self.new_patch_size[0]
        temp0 = x.reshape(
            B,
            self.in_dim,
            H // self.stride // new_patch_size,
            new_patch_size,
            W // self.stride // new_patch_size,
            new_patch_size,
        )  # [B, 24, 56, 56] -> [B, 24, 14, 4, 14, 4]
        temp0 = temp0.permute(
            0, 1, 3, 5, 2, 4
        )  # [B, 24, 14, 4, 14, 4] -> [B, 24, 4, 4, 14, 14]
        temp0 = temp0.reshape(
            B, self.in_dim * new_patch_size * new_patch_size, -1
        )  # [B, 24, 4, 4, 14, 14] -> [B, 384, 196]
        # [56] -> [4, 14]
        temp1 = x.reshape(
            B,
            self.in_dim,
            new_patch_size,
            H // self.stride // new_patch_size,
            new_patch_size,
            W // self.stride // new_patch_size,
        )  # [B, 24, 56, 56] -> [B, 24, 4, 14, 4, 14]
        temp1 = temp1.permute(
            0, 1, 2, 4, 3, 5
        )  # [B, 24, 4, 14, 4, 14] -> [B, 24, 4, 4, 14, 14]
        temp1 = temp1.reshape(
            B, self.in_dim * new_patch_size * new_patch_size, -1
        )  # [B, 24, 4, 4, 14, 14] -> [B, 384, 196]
        # print(temp0.eq(temp1).sum().item())  # 75264 = 1 * 384 * 196
        # -----------------------------------------------------------------------------------------------------------------------#

        x = self.unfold(
            x
        )  # [B, 24, 56, 56] -> [B, 384, 196]  56/4=14 14*14=196 24*4*4=384
        x = x.transpose(1, 2)  # [B, 384, 196] -> [B, 196, 384]
        # [B, 196, 384] -> [B*196, 24, 4, 4]
        x = x.reshape(
            B * self.num_patches,
            self.in_dim,
            self.new_patch_size[0],
            self.new_patch_size[1],
        )
        x = x + pixel_pos  # [B*196, 24, 4, 4] + [1, 24,  4,  4] = [B*196, 24, 4, 4]
        x = x.reshape(
            B * self.num_patches, self.in_dim, -1
        )  # [B*196, 24, 4, 4] -> [B*196, 24, 16]
        x = x.transpose(1, 2)  # [B*196, 24, 16] -> [B*196, 16, 24]
        return x


def test_reshape(shape: list[int] = [1, 24, 56, 56], new_patch_size: int = 4):
    """测试不同reshape和permute和Unfold的差别

    Args:
        shape (list[int], optional): Tensor形状. Defaults to [1, 24, 56, 56].
        new_patch_size (int, optional): 新的H,W. Defaults to 4.
    """
    B, C, H, W = shape
    x = torch.randn(shape)

    # [56] -> [14, 4]
    y1 = x.reshape(
        B, C, H // new_patch_size, new_patch_size, W // new_patch_size, new_patch_size
    )  # [B, 24, 56, 56] -> [B, 24, 14, 4, 14, 4]
    y1 = y1.permute(0, 1, 3, 5, 2, 4)  # [B, 24, 14, 4, 14, 4] -> [B, 24, 4, 4, 14, 14]
    y1 = y1.reshape(
        B, C * new_patch_size * new_patch_size, -1
    )  # [B, 24, 4, 4, 14, 14] -> [B, 384, 196]

    # [56] -> [4, 14]
    y2 = x.reshape(
        B, C, new_patch_size, H // new_patch_size, new_patch_size, W // new_patch_size
    )  # [B, 24, 56, 56] -> [B, 24, 4, 14, 4, 14]
    y2 = y2.permute(0, 1, 2, 4, 3, 5)  # [B, 24, 4, 14, 4, 14] -> [B, 24, 4, 4, 14, 14]
    y2 = y2.reshape(
        B, C * new_patch_size * new_patch_size, -1
    )  # [B, 24, 4, 4, 14, 14] -> [B, 384, 196]

    y3 = nn.Unfold(kernel_size=new_patch_size, stride=new_patch_size)(
        x
    )  # [B, 24, 56, 56] -> [B, 384, 196]
    print(y1.eq(y2).sum().item())  # 96
    print(y2.eq(y3).sum().item())  # 96
    print(y3.eq(y1).sum().item())  # 75264 = 1 * 384 * 196


# test_reshape()


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


# ---------------------------------------#
#   pixel_embed和patch_embed的Attn
# ---------------------------------------#
class Attention(nn.Module):
    """Multi-Head Attention"""

    def __init__(
        self, dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):  # pixel_embed               patch_embed
        B, N, C = x.shape  # [B*196, 16, 24]           [B, 197, 384]
        qk = self.qk(x)  # [B*196, 16, 48]           [B, 197, 768]
        qk = qk.reshape(
            B, N, 2, self.num_heads, self.head_dim
        )  # [B*196, 16, 2, 4, 6]      [B, 197, 2, 6, 64]
        qk = qk.permute(2, 0, 3, 1, 4)  # [2, b*196, 4, 16, 6]      [2, b, 6, 197, 64]
        q, k = qk.unbind(
            0
        )  # make torchscript happy (cannot use tensor as tuple) # [B*196, 4, 16, 6] * 2     [B, 6, 197, 64] * 2
        v = (
            self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        )  # [B*196, 4, 16, 6]         [B, 6, 197, 64]

        # [B*196, 4, 16, 6] @ [B*196, 4, 6, 16] = [B*196, 4, 16, 16]    [B, 6, 197, 64] @ [B, 6, 64, 197] = [B, 6, 197, 197]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (
            attn @ v
        )  # [B*196, 4, 16, 16] @ [B*196, 4, 16, 6] = [B*196, 4, 16, 6]    [B, 6, 197, 197] @ [B, 6, 197, 64] = [B, 6, 197, 64]
        x = x.transpose(
            1, 2
        ).reshape(
            B, N, -1
        )  # [B*196, 4, 16, 6] -> [B*196, 16, 4, 6] -> [B*196, 16, 24]     [B, 6, 197, 64] -> [B, 197, 6, 64] -> [B, 197, 384]
        x = self.proj(
            x
        )  # [B*196, 16, 24] -> [B*196, 16, 24]                            [B, 197, 384] -> [B, 197, 384]
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """TNT Block"""

    def __init__(
        self,
        dim,
        dim_out,
        num_pixel,
        num_heads_in=4,
        num_heads_out=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(dim)
        self.attn_in = Attention(
            dim,
            dim,
            num_heads=num_heads_in,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm_mlp_in = norm_layer(dim)
        self.mlp_in = Mlp(
            in_features=dim,
            hidden_features=int(dim * 4),
            out_features=dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.norm1_proj = norm_layer(dim)
        self.proj = nn.Linear(dim * num_pixel, dim_out, bias=True)

        # Outer transformer
        self.norm_out = norm_layer(dim_out)
        self.attn_out = Attention(
            dim_out,
            dim_out,
            num_heads=num_heads_out,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_mlp = norm_layer(dim_out)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            out_features=dim_out,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, pixel_embed, patch_embed):
        """
        Args:
            pixel_embed (Tensor): [B*position, 每个position的小patch数量, 每个小patch的channel]  ex:[B*196, 16, 24]
            patch_embed (Tensor): [B, position, channel] ex:[B, 197, 384]
        Returns:
            Tuple(Tensor): (pixel_embed, patch_embed) ex: ([B*196, 16, 24], [B, 197, 384])
        """
        # inner
        pixel_embed = pixel_embed + self.drop_path(
            self.attn_in(self.norm_in(pixel_embed))
        )  # [B*196, 16, 24] -> [B*196, 16, 24]
        pixel_embed = pixel_embed + self.drop_path(
            self.mlp_in(self.norm_mlp_in(pixel_embed))
        )  # [B*196, 16, 24] -> [B*196, 16, 24]

        # outer
        # patch_embed 全局注意力和mlp
        # 先将pixel_embed reshape成为和patch_embed相同的形状,拼接起来融合特征再进行全局注意力和mlp
        B, N, C = patch_embed.size()  # [B, 197, 384]
        temp_pixel_embed = self.norm1_proj(pixel_embed).reshape(
            B, N - 1, -1
        )  # [B*196, 16, 24] -> [B, 196, 384]
        temp_pixel_embed = self.proj(temp_pixel_embed)  # [B, 196, 384] -> [B, 196, 384]
        patch_embed = torch.cat(  # [B, 1, 384] cat ([B, 196, 384] + [B, 196, 384]) = [B, 197, 384]
            [patch_embed[:, 0:1], patch_embed[:, 1:] + temp_pixel_embed], dim=1
        )  # 取出分类层[B, 1, 384], 后面的位置[1, 196, 384]和pixel_embed相加然后再拼接到一起
        patch_embed = patch_embed + self.drop_path(
            self.attn_out(self.norm_out(patch_embed))
        )  # [B, 197, 384] -> [B, 197, 384]
        patch_embed = patch_embed + self.drop_path(
            self.mlp(self.norm_mlp(patch_embed))
        )  # [B, 197, 384] -> [B, 197, 384]
        return pixel_embed, patch_embed


class TNT(nn.Module):
    """Transformer in Transformer - https://arxiv.org/abs/2103.00112"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        inner_dim=48,
        depth=12,
        num_heads_inner=4,
        num_heads_outer=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        first_stride=4,
    ):
        super().__init__()
        assert global_pool in ("", "token", "avg")
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.grad_checkpointing = False

        self.pixel_embed = PixelEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            in_dim=inner_dim,
            stride=first_stride,
        )
        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size[0] * new_patch_size[1]

        self.norm1_proj = norm_layer(num_pixel * inner_dim)
        self.proj = nn.Linear(num_pixel * inner_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        # 分类参数 [1, 1, 384]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置参数 [1, 197, 384]
        self.patch_pos = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # pixel参数 [1, 24, 4, 4]
        self.pixel_pos = nn.Parameter(
            torch.zeros(1, inner_dim, new_patch_size[0], new_patch_size[1])
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(
                Block(
                    dim=inner_dim,
                    dim_out=embed_dim,
                    num_pixel=num_pixel,
                    num_heads_in=num_heads_inner,
                    num_heads_out=num_heads_outer,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.patch_pos, std=0.02)
        trunc_normal_(self.pixel_pos, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "token", "avg")
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        B = x.shape[0]  # [1, 3, 224, 224]
        # [B*position, 每个position的小patch数量, 每个小patch的channel] [1, 3, 224, 224] & [1, 24, 4, 4] -> [B*196, 16, 24]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)

        # [B*position, 每个position的小patch数量, 每个小patch的channel] -> [B, position, channel]   [B*196, 16, 24] -> [B, 196, 384]
        patch_embed = self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))
        patch_embed = self.norm2_proj(
            self.proj(patch_embed)
        )  # [B, 196, 384] -> [B, 196, 384]
        patch_embed = torch.cat(
            (self.cls_token.expand(B, -1, -1), patch_embed), dim=1
        )  # [B, 1, 384] cat [B, 196, 384] -> [B, 197, 384]
        patch_embed = (
            patch_embed + self.patch_pos
        )  # [B, 197, 384] + [B, 197, 384] = [B, 197, 384]
        patch_embed = self.pos_drop(patch_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for blk in self.blocks:
                pixel_embed, patch_embed = checkpoint(blk, pixel_embed, patch_embed)
        else:
            for blk in self.blocks:
                pixel_embed, patch_embed = blk(
                    pixel_embed, patch_embed
                )  # [B*196, 16, 24] & [B, 197, 384] -> [B*196, 16, 24] & [B, 197, 384]

        patch_embed = self.norm(patch_embed)
        return patch_embed

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, 1:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
            )  # 不是avg,所以是 x[:, 0] = [B, 384]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)  # [B, 384] -> [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)  # [1, 3, 224, 224] -> [B, 197, 384]
        x = self.forward_head(x)  # [B, 197, 384] -> [B, num_classes]
        return x


def checkpoint_filter_fn(state_dict, model):
    """convert patch embedding weight from manual patchify + linear proj to conv"""
    if state_dict["patch_pos"].shape != model.patch_pos.shape:
        state_dict["patch_pos"] = resize_pos_embed(
            state_dict["patch_pos"],
            model.patch_pos,
            getattr(model, "num_tokens", 1),
            model.pixel_embed.grid_size,
        )
    return state_dict


def _create_tnt(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    model = build_model_with_cfg(
        TNT, variant, pretrained, pretrained_filter_fn=checkpoint_filter_fn, **kwargs
    )
    return model


def tnt_s_patch16_224(pretrained=False, **kwargs) -> TNT:
    model_cfg = dict(
        patch_size=16,
        embed_dim=384,
        inner_dim=24,
        depth=12,
        num_heads_outer=6,
        qkv_bias=False,
    )
    model = _create_tnt(
        "tnt_s_patch16_224", pretrained=pretrained, **dict(model_cfg, **kwargs)
    )
    return model


def tnt_b_patch16_224(pretrained=False, **kwargs) -> TNT:
    model_cfg = dict(
        patch_size=16,
        embed_dim=640,
        inner_dim=40,
        depth=12,
        num_heads_outer=10,
        qkv_bias=False,
    )
    model = _create_tnt(
        "tnt_b_patch16_224", pretrained=pretrained, **dict(model_cfg, **kwargs)
    )
    return model


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = tnt_s_patch16_224(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    # 查看结构
    if False:
        onnx_path = "twins_pcpvt_small.onnx"
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
