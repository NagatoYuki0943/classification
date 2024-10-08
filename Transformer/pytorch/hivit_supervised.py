# https://github.com/zhangxiaosong18/hivit/blob/master/supervised/models/hivit.py

from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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


class Attention(nn.Module):
    def __init__(
        self,
        input_size,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rpe=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = (
            nn.Parameter(
                torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
            )
            if rpe
            else None
        )
        if rpe:
            coords_h = torch.arange(input_size)
            coords_w = torch.arange(input_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += input_size - 1
            relative_coords[:, :, 1] += input_size - 1
            relative_coords[:, :, 0] *= 2 * input_size - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape  # [B, 196, 512]
        qkv = self.qkv(x)  # [B, N, C] -> [B, N, 3*C]
        qkv = qkv.reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        )  # [B, N, 3*C] -> [B, N, 3, h, c]    C = h * c
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [B, N, 3, h, c] -> [3, B, h, N, c]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy      # [3, B, h, N, c] -> 3 * [B, h, N, c]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B, h, N, c] @ [B, h, c, N] = [B, h, N, N]

        if rpe_index is not None:
            rpe_index = self.relative_position_index.view(-1)
            S = int(math.sqrt(rpe_index.size(-1)))  # S = N
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(
                -1, S, S, self.num_heads
            )  # [1, N, N, h]
            relative_position_bias = relative_position_bias.permute(
                0, 3, 1, 2
            ).contiguous()  # [1, N, N, h] -> [1, h, N, N]
            attn = (
                attn + relative_position_bias
            )  # [B, h, N, N] + [1, h, N, N] = [B, h, N, N]
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = self.softmax(attn)  # 行上做softmax
        attn = self.attn_drop(attn)

        x = attn @ v  # [B, h, N, N] @ [B, h, N, c] = [B, h, N, c]
        x = x.transpose(1, 2)  # [B, h, N, c] -> [B, N, h, c]
        x = x.reshape(B, N, C)  # [B, N, h, c] -> [B, N, C]

        x = self.proj(x)  # [B, N, C] -> [B, N, C]
        x = self.proj_drop(x)
        return x


# --------------------#
#   重复的block
# --------------------#
class BlockWithRPE(nn.Module):
    def __init__(
        self,
        input_size,
        dim,
        num_heads=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        rpe=True,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        with_attn = num_heads > 0.0

        self.norm1 = norm_layer(dim) if with_attn else None
        self.attn = (
            Attention(
                input_size,
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                rpe=rpe,
            )
            if with_attn
            else None
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:  # stage1,2只有mlp,stage3才用attn
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# --------------------#
#   开始的stem
# --------------------#
class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        inner_patches=4,
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        kernel_size=None,
        pad_size=None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        kernel_size = kernel_size or conv_size
        pad_size = pad_size or 0
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=conv_size,
            padding=pad_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape  # [B, 3, 224, 224]
        patches_resolution = (
            H // self.patch_size[0],
            W // self.patch_size[1],
        )  # [14, 14]
        num_patches = patches_resolution[0] * patches_resolution[1]  # 14 * 14 = 196
        x = self.proj(x)  # [B, 3, 224, 224] -> [B, 128, 56, 56]
        x = x.view(  # [B, 128, 56, 56] -> [B, 128, 14, 4 , 14, 4]
            B,
            -1,
            patches_resolution[0],  # patches_resolution 代表windows个数
            self.inner_patches,
            patches_resolution[1],
            self.inner_patches,
        )
        x = x.permute(
            0, 2, 4, 3, 5, 1
        )  # [B, 128, 14, 4 , 14, 4] -> [B, 14, 14, 4, 4, 128]
        # [B, 14, 14, 4, 4, 128] -> [B, 196, 4, 4, 128]
        x = x.reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x


# --------------------#
#   stage2,3的下采样
# --------------------#
class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x, *args, **kwargs):
        is_main_stage = len(x.shape) == 3
        if is_main_stage:
            B, N, C = x.shape
            S = int(math.sqrt(N))
            x = x.reshape(B, S // 2, 2, S // 2, 2, C)
            x = x.permute(0, 1, 3, 2, 4, 5)
            x = x.reshape(B, -1, 2, 2, C)
            # [B, num_patches, inner_patches, inner_patches, C]
        x0 = x[
            ..., 0::2, 0::2, :
        ]  # [B, num_patches, inner_patches/2, inner_patches/2, C] 左上
        x1 = x[
            ..., 1::2, 0::2, :
        ]  # [B, num_patches, inner_patches/2, inner_patches/2, C] 左下
        x2 = x[
            ..., 0::2, 1::2, :
        ]  # [B, num_patches, inner_patches/2, inner_patches/2, C] 右上
        x3 = x[
            ..., 1::2, 1::2, :
        ]  # [B, num_patches, inner_patches/2, inner_patches/2, C] 右下

        x = torch.cat(
            [x0, x1, x2, x3], dim=-1
        )  # [B, num_patches, inner_patches/2, inner_patches/2, 512]
        x = self.norm(x)
        x = self.reduction(
            x
        )  # [B, num_patches, inner_patches/2, inner_patches/2, 4*C] -> [B, num_patches, inner_patches/2, inner_patches/2, 2*C]

        if is_main_stage:
            x = x[:, :, 0, 0, :]
        return x


class HiViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        inner_patches=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=512,
        depths=[4, 4, 19],
        num_heads=8,
        stem_mlp_ratio=3.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=True,
        rpe=True,
        patch_norm=True,
        use_checkpoint=False,
        kernel_size=None,
        pad_size=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        if self.num_layers <= 3:
            self.num_main_blocks = depths[-1]
        else:
            self.num_main_blocks = depths[2] + depths[3] + 1
            embed_dim *= 2

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            inner_patches=inner_patches,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            kernel_size=kernel_size,
            pad_size=pad_size,
        )
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1]))
        )

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_i, stage_depth in enumerate(depths):
            if stage_i == 3:
                Hp = Hp // 2
            is_main_stage = embed_dim == self.num_features or stage_i >= 2
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp,
                        embed_dim,
                        nhead,
                        ratio,
                        qkv_bias,
                        qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=next(dpr),
                        rpe=rpe,
                        norm_layer=norm_layer,
                    )
                )
            if stage_i + 1 < self.num_layers:
                self.blocks.append(PatchMerge(embed_dim, norm_layer))
                embed_dim *= 2
        if self.num_layers > 3:
            self.num_features *= 2
        self.fc_norm = norm_layer(self.num_features)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x, ids_keep=None, mask=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        if ids_keep is not None:
            x = torch.gather(
                x,
                dim=1,
                index=ids_keep[:, :, None, None, None].expand(-1, -1, *x.shape[2:]),
            )

        for blk in self.blocks[: -self.num_main_blocks]:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = x[..., 0, 0, :]
        if self.ape:
            pos_embed = self.absolute_pos_embed
            if ids_keep is not None:
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1),
                    dim=1,
                    index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),
                )
            x += pos_embed
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(
                    rpe_index[ids_keep, :],
                    dim=-1,
                    index=ids_keep[:, None, :].expand(-1, L, -1),
                ).reshape(B, -1)
            else:
                rpe_index = True

        for blk in self.blocks[-self.num_main_blocks :]:
            x = (
                checkpoint.checkpoint(blk, x, rpe_index, mask)
                if self.use_checkpoint
                else blk(x, rpe_index, mask)
            )

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)
        return x
