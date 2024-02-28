""" Nested Transformer (NesT) in PyTorch

A PyTorch implement of Aggregating Nested Transformers as described in:

'Aggregating Nested Transformers'
    - https://arxiv.org/abs/2105.12723

The official Jax code is released and available at https://github.com/google-research/nested-transformer. The weights
have been converted with convert/convert_nest_flax.py

Acknowledgments:
* The paper authors for sharing their research, code, and model weights
* Ross Wightman's existing code off which I based this

Copyright 2021 Alexander Soare
"""

import collections.abc
import logging
import math
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, create_classifier, trunc_normal_, _assert, to_2tuple
from timm.layers import create_conv2d, create_pool2d, to_ntuple, use_fused_attn, LayerNorm
from timm.layers.format import Format, nchw_to
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_function
from timm.models._manipulate import checkpoint_seq, named_apply
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations

__all__ = ['Nest']  # model_registry will add each entrypoint fn to this

_logger = logging.getLogger(__name__)


#-------------------------------------#
#   Patch BCHW -> BNC   out=[B, position, channel]
#   [B, 3, 224, 224] -> [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
#-------------------------------------#
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
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
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x)    # [B, 3, 224, 224] -> [B, 96, 56, 56]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


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


class Attention(nn.Module):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x is shape: B (batch_size), T (image blocks), N (seq length per image block), C (embed dim)
        """
        B, T, N, C = x.shape    # B, T, N, C   N = 196
        # result of next line is (qkv, B, num (H)eads, T, N, (C')hannels per head)
        qkv = self.qkv(x)       # [B, T, 196, C] -> [B, T, 196, C*3]        C = h * 32  per head has 32 channels
        qkv = qkv.reshape(B, T, N, 3, self.num_heads, C // self.num_heads)  # [B, T, 196, 96*3] -> [B, T, 196, 3, h, 32]
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)                                 # [B, T, 196, 3, h, 32] -> [3, B, h, T, 196, 32]
        q, k, v = qkv.unbind(0)  # make torchscript happy                     [3, B, h, T, 196, 32] -> [B, h, T, 196, 32] * 3

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # (B, H, T, N, N)    [B, h, T, 196, 32] @ [B, h, T, 32, 196] = [B, h, T, 196, 196]
            attn = attn.softmax(dim=-1)                         # 取每一列,在行上做softmax
            attn = self.attn_drop(attn)
            x = attn @ v                    # [B, h, T, 196, 196] @ [B, h, T, 196, 32] = [B, h, T, 196, 32]

        # (B, H, T, N, C'), permute -> (B, T, N, C', H)
        x = x.permute(0, 2, 3, 4, 1)        # [B, h, T, 196, 32] -> [B, T, 196, 32, h]
        x = x.reshape(B, T, N, C)           # [B, T, 196, 32, h] -> [B, T, 196, C]  C = h * 32

        x = self.proj(x)                    # [B, T, 196, C] -> [B, T, 196, C]
        x = self.proj_drop(x)
        return x  # (B, T, N, C)


class TransformerLayer(nn.Module):
    """
    This is much like `.vision_transformer.Block` but:
        - Called TransformerLayer here to allow for "block" as defined in the paper ("non-overlapping image blocks")
        - Uses modified Attention layer that handles the "block" dimension
    """
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


#----------------------------------#
#   下采样 宽高减半
#   3x3Conv LN Pool(k=3 s=2 p=1)
#----------------------------------#
class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, pad_type=''):
        super().__init__()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size=3, padding=pad_type, bias=True)
        self.norm = norm_layer(out_channels)
        self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        _assert(x.shape[-2] % 2 == 0, 'BlockAggregation requires even input spatial dims')
        _assert(x.shape[-1] % 2 == 0, 'BlockAggregation requires even input spatial dims')
        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        return x  # (B, C, H//2, W//2)


#------------------------------------------#
#   将宽高划分为patch
#   [B, 56, 56, 96]  -> [B, 16, 196, 96]
#   [B, 28, 28, 192] -> [B, 4, 196, 192]
#   [B, 14, 14, 384] -> [B, 1, 196, 384]
#------------------------------------------#
def blockify(x, block_size: int):
    """image to blocks
    Args:
        x (Tensor): with shape (B, H, W, C)
        block_size (int): edge length of a single square block in units of H, W
    """
    B, H, W, C  = x.shape           # [B, 56, 56, 96]
    _assert(H % block_size == 0, '`block_size` must divide input height evenly')
    _assert(W % block_size == 0, '`block_size` must divide input width evenly')
    grid_height = H // block_size   # 56 / 14 = 4
    grid_width  = W // block_size   # 56 / 14 = 4
    x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)    # [B, 56, 56, 96] -> [B, 4, 14, 4, 14, 96]
    x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)       # [B, 4, 14, 4, 14, 96] -> [B, 4, 4, 14, 14, 96] -> [B, 16, 196, 96]
    return x  # (B, T, N, C)         [B, 16, 196, 96]


#------------------------------------------#
#   将划分为patch的宽高还原
#   [B, 16, 196, 96] -> [B, 56, 56, 96]
#   [B, 4, 196, 192] -> [B, 28, 28, 192]
#   [B, 1, 196, 384] -> [B, 14, 14, 384]
#------------------------------------------#
@register_notrace_function  # reason: int receives Proxy
def deblockify(x, block_size: int):
    """blocks to image
    Args:
        x (Tensor): with shape (B, T, N, C) where T is number of blocks and N is sequence size per block
        block_size (int): edge length of a single square block in units of desired H, W
    """
    B, T, _, C = x.shape                    # [B, 16, 196, 96]
    grid_size = int(math.sqrt(T))           # sqrt(16) = 4
    height = width = grid_size * block_size # 4 * 14 = 56
    x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x = x.transpose(2, 3).reshape(B, height, width, C)  # [B, 16, 196, 96] -> [B, 16, 96, 196] -> [B, 56, 56, 96]
    return x  # (B, H, W, C)


#-------------------------------------#
#   NesT的3个level
#   开始会pool降采样
#   每个level内部会有n次Transformer
#-------------------------------------#
class NestLevel(nn.Module):
    """ Single hierarchical level of a Nested Transformer
    """
    def __init__(
            self,
            num_blocks,
            block_size,
            seq_length,
            num_heads,
            depth,
            embed_dim,
            prev_embed_dim=None,
            mlp_ratio=4.,
            qkv_bias=True,
            proj_drop=0.,
            attn_drop=0.,
            drop_path=[],
            norm_layer=None,
            act_layer=None,
            pad_type='',
    ):
        super().__init__()
        self.block_size = block_size
        self.grad_checkpointing = False

        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))

        if prev_embed_dim is not None:  # 3x3Conv LN Pool(k=3 s=2 p=1)
            self.pool = ConvPool(prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type)
        else:
            self.pool = nn.Identity()   # 第一次是这个

        # Transformer encoder
        if len(drop_path):
            assert len(drop_path) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.transformer_encoder = nn.Sequential(*[
            TransformerLayer(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        """
        expects x as (B, C, H, W)
        """                                 # pool               Identity           k=3 s=2 p=1         k=3 s=2 p=1
        x = self.pool(x)                    # [B, 96, 56, 56] -> [B, 96, 56, 56] -> [B, 192, 28, 28] -> [B, 384, 14, 14]
        x = x.permute(0, 2, 3, 1)           #                    [B, 56, 56, 96]    [B, 28, 28, 192]    [B, 14, 14, 384]
        x = blockify(x, self.block_size)    #                    [B, 16, 196, 96]   [B, 4, 196, 192]    [B, 1, 196, 384]
        x = x + self.pos_embed              #                    [B, 16, 196, 96]   [B, 4, 196, 192]    [B, 1, 196, 384]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.transformer_encoder, x)
        else:
            x = self.transformer_encoder(x)                    # [B, 16, 196, 96]   [B, 4, 196, 192]    [B, 1, 196, 384]
        x = deblockify(x, self.block_size)                     # [B, 56, 56, 96]    [B, 28, 28, 192]    [B, 14, 14, 384]
        # Channel-first for block aggregation, and generally to replicate convnet feature map at each stage
        return x.permute(0, 3, 1, 2)  # (B, C, H', W')           [B, 96, 56, 56]    [B, 192, 28, 28]    [B, 384, 14, 14]


class Nest(nn.Module):
    """ Nested Transformer (NesT)

    A PyTorch impl of : `Aggregating Nested Transformers`
        - https://arxiv.org/abs/2105.12723
    """

    def __init__(
            self,
            img_size=224,
            in_chans=3,
            patch_size=4,
            num_levels=3,
            embed_dims=(128, 256, 512),
            num_heads=(4, 8, 16),
            depths=(2, 2, 20),
            num_classes=1000,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.5,
            norm_layer=None,
            act_layer=None,
            pad_type='',
            weight_init='',
            global_pool='avg',
    ):
        """
        Args:
            img_size (int, tuple):   input image size
            in_chans (int):          number of input channels
            patch_size (int):        patch size
            num_levels (int):        number of block hierarchies (T_d in the paper)
            embed_dims (int, tuple): embedding dimensions of each level
            num_heads (int, tuple):  number of attention heads for each level
            depths (int, tuple):     number of transformer layers for each level
            num_classes (int):       number of classes for classification head
            mlp_ratio (int):         ratio of mlp hidden dim to embedding dim for MLP of transformer layers
            qkv_bias (bool):         enable bias for qkv if True
            drop_rate (float):       dropout rate for MLP of transformer layers, MSA final projection layer, and classifier
            attn_drop_rate (float):  attention dropout rate
            drop_path_rate (float):  stochastic depth rate
            norm_layer (nn.Module):  normalization layer for transformer layers
            act_layer (nn.Module):   activation layer in MLP of transformer layers
            pad_type (str):          Type of padding to use '' for PyTorch symmetric, 'same' for TF SAME
            weight_init (str):       weight init scheme
            global_pool (str):       type of pooling operation to apply to final feature map

        Notes:
            - Default values follow NesT-B from the original Jax code.
            - `embed_dims`, `num_heads`, `depths` should be ints or tuples with length `num_levels`.
            - For those following the paper, Table A1 may have errors!
                - https://github.com/google-research/nested-transformer/issues/2
        """
        super().__init__()

        for param_name in ['embed_dims', 'num_heads', 'depths']:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == num_levels, f'Require `len({param_name}) == num_levels`'

        embed_dims = to_ntuple(num_levels)(embed_dims)
        num_heads = to_ntuple(num_levels)(num_heads)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.feature_info = []
        norm_layer = norm_layer or LayerNorm
        act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate
        self.num_levels = num_levels
        if isinstance(img_size, collections.abc.Sequence):
            assert img_size[0] == img_size[1], 'Model only handles square inputs'
            img_size = img_size[0]
        assert img_size % patch_size == 0, '`patch_size` must divide `img_size` evenly'
        self.patch_size = patch_size

        # Number of blocks at each level
        self.num_blocks = (4 ** torch.arange(num_levels)).flip(0).tolist()
        assert (img_size // patch_size) % math.sqrt(self.num_blocks[0]) == 0, \
            'First level blocks don\'t fit evenly. Check `img_size`, `patch_size`, and `num_levels`'

        # Block edge size in units of patches
        # Hint: (img_size // patch_size) gives number of patches along edge of image. sqrt(self.num_blocks[0]) is the
        #  number of blocks along edge of image
        self.block_size = int((img_size // patch_size) // math.sqrt(self.num_blocks[0]))

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            flatten=False,
        )
        self.num_patches = self.patch_embed.num_patches
        self.seq_length = self.num_patches // self.num_blocks[0]

        # Build up each hierarchical level
        levels = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        curr_stride = 4
        for i in range(len(self.num_blocks)):
            dim = embed_dims[i]
            levels.append(NestLevel(
                self.num_blocks[i],
                self.block_size,
                self.seq_length,
                num_heads[i],
                depths[i],
                dim,
                prev_dim,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dp_rates[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                pad_type=pad_type,
            ))
            self.feature_info += [dict(num_chs=dim, reduction=curr_stride, module=f'levels.{i}')]
            prev_dim = dim
            curr_stride *= 2
        self.levels = nn.Sequential(*levels)

        # Final normalization layer
        self.norm = norm_layer(embed_dims[-1])

        # Classifier
        global_pool, head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        self.global_pool = global_pool
        self.head_drop = nn.Dropout(drop_rate)
        self.head = head

        self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        for level in self.levels:
            trunc_normal_(level.pos_embed, std=.02, a=-2, b=2)
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)                                     # [B, 3, 224, 224] -> [B, 96, 56, 56]
        x = self.levels(x)                                          # [B, 96, 56, 56] -> [B, 96, 56, 56] -> [B, 192, 28, 28] -> [B, 384, 14, 14]
        # Layer norm done over channel dim only (to NHWC and back)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    # [B, 384, 14, 14] -> [B, 14, 14, 384] -> [B, 384, 14, 14]
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)                     # [B, 384, 14, 14] -> [B, 384]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)    # [B, 384] -> [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_nest_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02, a=-2, b=2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def resize_pos_embed(posemb, posemb_new):
    """
    Rescale the grid of position embeddings when loading from state_dict
    Expected shape of position embeddings is (1, T, N, C), and considers only square images
    """
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    seq_length_old = posemb.shape[2]
    num_blocks_new, seq_length_new = posemb_new.shape[1:3]
    size_new = int(math.sqrt(num_blocks_new*seq_length_new))
    # First change to (1, C, H, W)
    posemb = deblockify(posemb, int(math.sqrt(seq_length_old))).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=[size_new, size_new], mode='bicubic', align_corners=False)
    # Now change to new (1, T, N, C)
    posemb = blockify(posemb.permute(0, 2, 3, 1), int(math.sqrt(seq_length_new)))
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ resize positional embeddings of pretrained weights """
    pos_embed_keys = [k for k in state_dict.keys() if k.startswith('pos_embed_')]
    for k in pos_embed_keys:
        if state_dict[k].shape != getattr(model, k).shape:
            state_dict[k] = resize_pos_embed(state_dict[k], getattr(model, k))
    return state_dict


def _create_nest(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        Nest,
        variant,
        pretrained,
        feature_cfg=dict(out_indices=(0, 1, 2), flatten_sequential=True),
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )

    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': [14, 14],
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'nest_base.untrained': _cfg(),
    'nest_small.untrained': _cfg(),
    'nest_tiny.untrained': _cfg(),
    # (weights from official Google JAX impl, require 'SAME' padding)
    'nest_base_jx.goog_in1k': _cfg(hf_hub_id='timm/'),
    'nest_small_jx.goog_in1k': _cfg(hf_hub_id='timm/'),
    'nest_tiny_jx.goog_in1k': _cfg(hf_hub_id='timm/'),
})


def nest_base(pretrained=False, **kwargs) -> Nest:
    """ Nest-B @ 224x224
    """
    model_kwargs = dict(
        embed_dims=(128, 256, 512), num_heads=(4, 8, 16), depths=(2, 2, 20), **kwargs)
    model = _create_nest('nest_base', pretrained=pretrained, **model_kwargs)
    return model


def nest_small(pretrained=False, **kwargs) -> Nest:
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384), num_heads=(3, 6, 12), depths=(2, 2, 20), **kwargs)
    model = _create_nest('nest_small', pretrained=pretrained, **model_kwargs)
    return model


def nest_tiny(pretrained=False, **kwargs) -> Nest:
    """ Nest-T @ 224x224
    """
    model_kwargs = dict(embed_dims=(96, 192, 384), num_heads=(3, 6, 12), depths=(2, 2, 8), **kwargs)
    model = _create_nest('nest_tiny', pretrained=pretrained, **model_kwargs)
    return model


def nest_base_jx(pretrained=False, **kwargs) -> Nest:
    """ Nest-B @ 224x224
    """
    kwargs.setdefault('pad_type', 'same')
    model_kwargs = dict(
        embed_dims=(128, 256, 512), num_heads=(4, 8, 16), depths=(2, 2, 20), **kwargs)
    model = _create_nest('nest_base_jx', pretrained=pretrained, **model_kwargs)
    return model


def nest_small_jx(pretrained=False, **kwargs) -> Nest:
    """ Nest-S @ 224x224
    """
    kwargs.setdefault('pad_type', 'same')
    model_kwargs = dict(embed_dims=(96, 192, 384), num_heads=(3, 6, 12), depths=(2, 2, 20), **kwargs)
    model = _create_nest('nest_small_jx', pretrained=pretrained, **model_kwargs)
    return model


def nest_tiny_jx(pretrained=False, **kwargs) -> Nest:
    """ Nest-T @ 224x224
    """
    kwargs.setdefault('pad_type', 'same')
    model_kwargs = dict(embed_dims=(96, 192, 384), num_heads=(3, 6, 12), depths=(2, 2, 8), **kwargs)
    model = _create_nest('nest_tiny_jx', pretrained=pretrained, **model_kwargs)
    return model


register_model_deprecations(__name__, {
    'jx_nest_base': 'nest_base_jx',
    'jx_nest_small': 'nest_small_jx',
    'jx_nest_tiny': 'nest_tiny_jx',
})


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = nest_base_jx(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]

    # 查看结构
    if False:
        onnx_path = 'nest_base_jx.onnx'
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
