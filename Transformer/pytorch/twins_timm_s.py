""" Twins
A PyTorch impl of : `Twins: Revisiting the Design of Spatial Attention in Vision Transformers`
    - https://arxiv.org/pdf/2104.13840.pdf

Code/weights from https://github.com/Meituan-AutoML/Twins, original copyright/license info below

"""
# --------------------------------------------------------
# Twins
# Copyright (c) 2021 Meituan
# Licensed under The Apache 2.0 License [see LICENSE for details]
# Written by Xinjie Li, Xiangxiang Chu
# --------------------------------------------------------
import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, to_2tuple, trunc_normal_, use_fused_attn
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_module
from timm.models._registry import register_model, generate_default_cfgs

__all__ = ['Twins']  # model_registry will add each entrypoint fn to this

Size_ = Tuple[int, int]


#-------------------------------------------------------------------------------------------------#
#   开始的patch以及下采样
#   [B, 3, 224, 224] -> [B, 64, 56, 56] -> [B, 128, 28, 28] -> [B, 256, 14, 14] -> [B, 512, 7, 7]
#                       [B, H*W, 64]  -> [B, 28*28, 128]  -> [B, 14*14, 256]  -> [B, 7*7, 512]
#-------------------------------------------------------------------------------------------------#
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, Size_]:
        """
        Args:
            x (Tensor): 图片 [B, C, H, W]

        Returns:
            Tuple[torch.Tensor, Size_]: 经过patch并调整为3维的图片 [B, N, C] 和新图片的高宽 [new_H, new_W]
        """
        B, C, H, W = x.shape
        # [B, 3, 224, 224] -> [B, 64, H, W] -> [B, 64, H*W] -> [B, H*W, 64]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        out_size = (H // self.patch_size[0], W // self.patch_size[1])

        return x, out_size


#-----------------------------------------#
#   [..., C] -> [..., n*C] -> [..., C]
#-----------------------------------------#
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
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
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):   # mix channel
        x = self.fc1(x)     # [B, N, C] -> [B, N, n*C]
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)     # [B, N, n*C] -> [B, N, C]
        x = self.drop2(x)
        return x


#-------------------------#
#   vit中的attn
#-------------------------#
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape   # [B, 197, 768] [B, position, channel]
        qkv = self.qkv(x)                                           # [B, 197, 768] -> [B, 197, 3 * 768]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)   # [B, 197, 3 * 768]   -> [B, 197, 3, 12, 64]
        qkv = qkv.permute(2, 0, 3, 1, 4)                            # [B, 197, 3, 12, 64] -> [3, B, 12, 197, 64]
        q, k, v = qkv.unbind(0)                                     # [3, B, 12, 197, 64] -> [B, 12, 197, 64] * 3
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, 12, 197, 64]  @ [B, 12, 64, 197] = [B, 12, 197, 197]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v                    # [B, 12, 197, 197] @ [B, 12, 197, 64] = [B, 12, 197, 64]

        x = x.transpose(1, 2)               # [B, 12, 197, 64] -> [B, 197, 12, 64]
        x = x.reshape(B, N, C)              # [B, 197, 12, 64] -> [B, 197, 768]

        x = self.proj(x)                    # [B, 197, 768] -> [B, 197, 768]
        x = self.proj_drop(x)
        return x


#-------------------------#
#   类似swin的做法
#   window size = 7
#-------------------------#
@register_notrace_module  # reason: FX can't symbolically trace control flow in forward method
class LocallyGroupedAttn(nn.Module):
    """ LSA: self attention within a group
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        """
        Args:
            x (Tensor): [B, N, C]
            size (Size_): [H, W]  H * W = P

        Returns:
            Tensor: [B, N, C]
        """
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape       # [B, 56*56, 64]
        H, W = size             # 56, 56
        x = x.view(B, H, W, C)  # [B, 56, 56, 64]

        # 在右下填充0,补全形状
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # 分离出小window
        _, Hp, Wp, _ = x.shape                          # 56, 56 新的宽高
        _h, _w = Hp // self.ws, Wp // self.ws           # 8 8 宽高划分为7的窗口,划分为8个
        x = x.reshape(B, _h, self.ws, _w, self.ws, C)   # [B, 56, 56, 64] -> [B, 8, 7, 8, 7, 64]
        x = x.transpose(2, 3)                           # [B, 8, 7, 8, 7, 64] -> [B, 8, 8, 7, 7, 64]

        qkv = self.qkv(x)                                               # [B, 8, 8, 7, 7, 64] -> [B, 8, 8, 7, 7, 64*3]
        qkv = qkv.reshape(B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads)  # [B, 8, 8, 7, 7, 64*3] -> [B, 64, 49, 3, 2, 32]
        qkv = qkv.permute(3, 0, 1, 4, 2, 5)                             # [B, 64, 49, 3, 2, 32] -> [3, B, 64, 2, 49, 32]
        q, k, v = qkv.unbind(0)                                         # [3, B, 64, 2, 49, 32] -> [B, 64, 2, 49, 32] * 3

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, 64, 2, 49, 32] @ [B, 64, 2, 32, 49] -> [B, 64, 2, 49, 49]
            attn = attn.softmax(dim=-1)     # 取每一列,在行上做softmax
            attn = self.attn_drop(attn)
            x = attn @ v                    # [B, 64, 2, 49, 49] @ [B, 64, 2, 49, 32] = [B, 64, 2, 49, 32]

        # 将分离出的小window还原
        x = x.transpose(2, 3)                           # [B, 64, 2, 49, 32] -> [B, 64, 49, 2, 32]
        x = x.reshape(B, _h, _w, self.ws, self.ws, C)   # [B, 64, 49, 2, 32] -> [B, 8, 8, 7, 7, 64]
        x = x.transpose(2, 3)                           # [B, 8, 8, 7, 7, 64] -> [B, 8, 7, 8, 7, 64]
        x = x.reshape(B, _h * self.ws, _w * self.ws, C) # [B, 8, 7, 8, 7, 64] -> [B, 56, 56, 64]

        # 将填充的部分删除
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)  # [B, 56, 56, 64] -> [B, H*W, 64]
        x = self.proj(x)        # [B, H*W, 64] -> [B, H*W, 64]
        x = self.proj_drop(x)
        return x

    # def forward_mask(self, x, size: Size_):
    #     B, N, C = x.shape
    #     H, W = size
    #     x = x.view(B, H, W, C)
    #     pad_l = pad_t = 0
    #     pad_r = (self.ws - W % self.ws) % self.ws
    #     pad_b = (self.ws - H % self.ws) % self.ws
    #     x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    #     _, Hp, Wp, _ = x.shape
    #     _h, _w = Hp // self.ws, Wp // self.ws
    #     mask = torch.zeros((1, Hp, Wp), device=x.device)
    #     mask[:, -pad_b:, :].fill_(1)
    #     mask[:, :, -pad_r:].fill_(1)
    #
    #     x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)  # B, _h, _w, ws, ws, C
    #     mask = mask.reshape(1, _h, self.ws, _w, self.ws).transpose(2, 3).reshape(1,  _h * _w, self.ws * self.ws)
    #     attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)  # 1, _h*_w, ws*ws, ws*ws
    #     attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0))
    #     qkv = self.qkv(x).reshape(
    #         B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
    #     # n_h, B, _w*_h, nhead, ws*ws, dim
    #     q, k, v = qkv[0], qkv[1], qkv[2]  # B, _h*_w, n_head, ws*ws, dim_head
    #     attn = (q @ k.transpose(-2, -1)) * self.scale  # B, _h*_w, n_head, ws*ws, ws*ws
    #     attn = attn + attn_mask.unsqueeze(2)
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)  # attn @v ->  B, _h*_w, n_head, ws*ws, dim_head
    #     attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
    #     x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
    #     if pad_r > 0 or pad_b > 0:
    #         x = x[:, :H, :W, :].contiguous()
    #     x = x.reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x


#---------------------------------------------------------#
#   q = q(x)
#   kv = kv(conv(x)) conv是patch,降低x的宽高,提取全局特征
#---------------------------------------------------------#
class GlobalSubSampleAttn(nn.Module):
    """ GSA: using a  key to summarize the information for a group to be efficient.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            #---------------------------------------------------------#
            #   对x处理,用来当做 key value 的输入
            #   kernel和stride相同 x的HW总为7,因为sr_ratio为 8 4 2 1 最后的1不处理
            #---------------------------------------------------------#
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        """
        Args:
            x (Tensor): [B, N, C]
            size (Size_): [H, W]  H * W = P

        Returns:
            Tensor: [B, N, C]
        """
        B, N, C = x.shape   # [B, H*W, C]

        # 只计算query  h * c = C
        # [B, N, C] -> [B, N, C] -> [B, N, h, c]    pcpvt每个head的channel总为64,svt中为32
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)   # [B, N, h, c] -> [B, h, N, c]

        #--------------------------------------------#
        #   对x处理,减少宽高,用来当做 key value 的输入
        #--------------------------------------------#
        if self.sr is not None:
            x = x.permute(0, 2, 1)      # [B, N, C] -> [B, C, N]
            x = x.reshape(B, C, *size)  # [B, C, N] -> [B, C, H, W]

            x = self.sr(x)              # [B, C, H, W] -> [B, C, 7, 7]  一个卷积,kernel和stride相同 x的HW总为7,因为sr_ratio为 8 4 2 1
            x = x.reshape(B, C, -1)     # [B, C, 7, 7] -> [B, C, 49]
            x = x.permute(0, 2, 1)      # [B, C, 49] -> [B, 49, C]
            x = self.norm(x)

        kv = self.kv(x)                                                 # [B, 49, C] -> [B, 49, 2*C]
        kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads)  # [B, 49, 2*C] -> [B, 49, 2, h, c]
        kv = kv.permute(2, 0, 3, 1, 4)                                  # [B, 49, 2, h, c] -> [2, B, h, 49, c]
        k, v = kv.unbind(0)                                             # [2, B, h, 49, c] -> [B, h, 49, c] * 2

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, h, N, c] @ [B, h, c, 49] = [B, h, N, 49]
            attn = attn.softmax(dim=-1)     # 取每一列,在行上做softmax
            attn = self.attn_drop(attn)
            x = attn @ v                    # [B, h, N, 49] @ [B, h, 49, c] = [B, h, N, c]

        x = x.transpose(1, 2)               # [B, h, N, c] -> [B, N, h, c]
        x = x.reshape(B, N, C)              # [B, N, h, c] -> [B, N, C]

        x = self.proj(x)            # [B, N, C] -> [B, N, C]
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            ws=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, proj_drop)
        elif ws == 1:
            self.attn = GlobalSubSampleAttn(dim, num_heads, attn_drop, proj_drop, sr_ratio)
        else:
            self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, proj_drop, ws)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, size: Size_):
        x = x + self.drop_path1(self.attn(self.norm1(x), size)) # [B, N, C] -> [B, N, C]
        x = x + self.drop_path2(self.mlp(self.norm2(x)))        # [B, N, C] -> [B, N, 4*C] -> [B, N, C]
        return x


#----------------------------------#
#   位置编码器
#   深度可分离卷积对[B, N, C]处理
#----------------------------------#
class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        #-----------------------------#
        #   深度可分离卷积,生成位置编码
        #-----------------------------#
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim),
        )
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)       # [B, C, H, W] -> [B, C, H, W]  深度可分离卷积,生成位置编码
        if self.stride == 1:
            x += cnn_feat_token             # [B, C, H, W] + [B, C, H, W] = [B, C, H, W] 位置编码和原数据相加
        x = x.flatten(2).transpose(1, 2)    # [B, C, H, W] -> [B, C, N] -> [B, N, C]
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class Twins(nn.Module):
    """ Twins Vision Transfomer (Revisiting Spatial Attention)

    Adapted from PVT (PyramidVisionTransformer) class at https://github.com/whai362/PVT.git
    """
    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            embed_dims=(64, 128, 256, 512),
            num_heads=(1, 2, 4, 8),
            mlp_ratios=(4, 4, 4, 4),
            depths=(3, 4, 6, 3),
            sr_ratios=(8, 4, 2, 1),
            wss=None,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            block_cls=Block,
    ):
        """
        Args:
            img_size (int, optional):           输入图片大小. Defaults to 224.
            patch_size (int, optional):         初始的patch_size. Defaults to 4.
            in_chans (int, optional):           图片输入通道. Defaults to 3.
            num_classes (int, optional):        分类数. Defaults to 1000.
            global_pool (str, optional):        使用哪种方式做最后的处理. Defaults to 'avg'.
            embed_dims (tuple, optional):       四个阶段的embed通道数. Defaults to (64, 128, 256, 512).
            num_heads (tuple, optional):        四个阶段的分头数. Defaults to (1, 2, 4, 8).
            mlp_ratios (tuple, optional):       四个阶段的mlp扩展倍率. Defaults to (4, 4, 4, 4).
            depths (tuple, optional):           四个阶段的block重复次数. Defaults to (3, 4, 6, 3).
            sr_ratios (tuple, optional):        四个阶段的Global中处理x的kernel和stride. Defaults to (8, 4, 2, 1).
            wss (list, optional):               window size list. Defaults to None.
            drop_rate (float, optional):        attn中的proj_drop和mlp中的drop. Defaults to 0..
            pos_drop_rate (float, optional):    patchembed后面的dropout rate. Defaults to 0..
            proj_drop_rate (float, optional):   mlp中的dropout rate. Defaults to 0..
            attn_drop_rate (float, optional):   attn中的attn_drop. Defaults to 0..
            drop_path_rate (float, optional):   stochastic depth decay drop rate. Defaults to 0..
            norm_layer (Module, optional):      Norm. Defaults to partial(nn.LayerNorm, eps=1e-6).
            block_cls (Module, optional):       使用的Block. Defaults to Block.
        """
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.grad_checkpointing = False

        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=pos_drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                dim=embed_dims[k],
                num_heads=num_heads[k],
                mlp_ratio=mlp_ratios[k],
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[k],
                ws=1 if wss is None or i % 2 == 1 else wss[k]) for i in range(depths[k])],
            )
            # ws=None,使用Attention; ws=1,使用GlobalSubSampleAttn; 否则使用LocallyGroupedAttn
            # pcpvt一直使用GlobalSubSampleAtt,而svt交替使用LocallyGroupedAttn和GlobalSubSampleAttn
            self.blocks.append(_block)
            cur += depths[k]

        self.pos_block = nn.ModuleList([PosConv(embed_dim, embed_dim) for embed_dim in embed_dims])

        self.norm = norm_layer(self.num_features)

        # classification head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]  # [B, 3, 224, 224]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x, size = embed(x)  # [B, 3, 224, 224] -> [B, N, C] & [H, W]  C = H * W
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)          # [B, N, C] -> [B, N, C]
                if j == 0:
                    x = pos_blk(x, size)  # PEG here 使用深度可分离卷积处理建立位置编码 [B, N, C] -> [B, N, C]
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()    # [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)                       # [B, 49, 512] -> [B, 512]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)    # [B, 512] -> [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)    # [B, 3, 224, 224] -> [B, 49, 512]
        x = self.forward_head(x)        # [B, 49, 512] -> [B, num_classes]
        return x


def _create_twins(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(Twins, variant, pretrained, **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embeds.0.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'twins_pcpvt_small.in1k': _cfg(hf_hub_id='timm/'),
    'twins_pcpvt_base.in1k': _cfg(hf_hub_id='timm/'),
    'twins_pcpvt_large.in1k': _cfg(hf_hub_id='timm/'),
    'twins_svt_small.in1k': _cfg(hf_hub_id='timm/'),
    'twins_svt_base.in1k': _cfg(hf_hub_id='timm/'),
    'twins_svt_large.in1k': _cfg(hf_hub_id='timm/'),
})


def twins_pcpvt_small(pretrained=False, **kwargs) -> Twins:
    """只使用GlobalSubSampleAttn
    """
    model_args = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1])
    return _create_twins('twins_pcpvt_small', pretrained=pretrained, **dict(model_args, **kwargs))


def twins_pcpvt_base(pretrained=False, **kwargs) -> Twins:
    """只使用GlobalSubSampleAttn
    """
    model_args = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1])
    return _create_twins('twins_pcpvt_base', pretrained=pretrained, **dict(model_args, **kwargs))


def twins_pcpvt_large(pretrained=False, **kwargs) -> Twins:
    """只使用GlobalSubSampleAttn
    """
    model_args = dict(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1])
    return _create_twins('twins_pcpvt_large', pretrained=pretrained, **dict(model_args, **kwargs))


def twins_svt_small(pretrained=False, **kwargs) -> Twins:
    """交替使用LocallyGroupedAttn和GlobalSubSampleAttn
        depths[0]=2,第1次使用LocallyGroupedAttn,第二次使用GlobalSubSampleAttn
    """
    model_args = dict(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1])
    return _create_twins('twins_svt_small', pretrained=pretrained, **dict(model_args, **kwargs))


def twins_svt_base(pretrained=False, **kwargs) -> Twins:
    """交替使用LocallyGroupedAttn和GlobalSubSampleAttn
        depths[0]=2,第1次使用LocallyGroupedAttn,第二次使用GlobalSubSampleAttn
    """
    model_args = dict(
        patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1])
    return _create_twins('twins_svt_base', pretrained=pretrained, **dict(model_args, **kwargs))


def twins_svt_large(pretrained=False, **kwargs) -> Twins:
    """交替使用LocallyGroupedAttn和GlobalSubSampleAttn
        depths[0]=2,第1次使用LocallyGroupedAttn,第二次使用GlobalSubSampleAttn
    """
    model_args = dict(
        patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1])
    return _create_twins('twins_svt_large', pretrained=pretrained, **dict(model_args, **kwargs))


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = twins_svt_small(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]

    # 查看结构
    if False:
        onnx_path = 'twins_svt_small.onnx'
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=['images'],
            output_names=['classes'],
        )
        import onnx
        from onnxsim import simplify
        # 载入onnx模型
        model_ = onnx.load(onnx_path)

        # 简化模型
        model_simple, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simple, onnx_path)
        print('finished exporting ' + onnx_path)
