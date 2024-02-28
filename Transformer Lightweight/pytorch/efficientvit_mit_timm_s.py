""" EfficientViT (by MIT Song Han's Lab)

Paper: `Efficientvit: Enhanced linear attention for high-resolution low-computation visual recognition`
    - https://arxiv.org/abs/2205.14756

Adapted from official impl at https://github.com/mit-han-lab/efficientvit
"""

__all__ = ['EfficientVit']
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, create_conv2d, GELUTanh
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_module
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs


def val2list(x: list or tuple or any, repeat_time=1):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1):
    # repeat elements if necessary
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


#---------------------#
#   conv前有dropout
#---------------------#
class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        dropout=0.,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.ReLU,
    ):
        super(ConvNormAct, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm_layer(num_features=out_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True) if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.dropout(x) # conv前有dropout
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


#--------------------------#
#   stem中使用
#   3x3DWConv + 1x1PWConv
#--------------------------#
class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, None),
    ):
        super(DSConv, self).__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)

        self.depth_conv = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.point_conv = ConvNormAct(
            in_channels,
            out_channels,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


#-----------------------#
#   stem中使用
#   3x3Conv + 3x3Conv
#-----------------------#
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, None),
    ):
        super(ConvBlock, self).__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvNormAct(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.conv2 = ConvNormAct(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


#---------------------------------------#
#   倒残差模块,EfficientVit[Large]使用
#   1x1PWConv + 3x3DWConv + 1x1PWConv
#---------------------------------------#
class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, nn.ReLU6, None),
    ):
        super(MBConv, self).__init__()
        use_bias = val2tuple(use_bias, 3)
        norm_layer = val2tuple(norm_layer, 3)
        act_layer = val2tuple(act_layer, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        # 1x1PWConv
        self.inverted_conv = ConvNormAct(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        # 3x3DWConv
        self.depth_conv = ConvNormAct(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )
        # 1x1PWConv
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[2],
            act_layer=act_layer[2],
            bias=use_bias[2],
        )

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


#-----------------------------------------#
#   Fused倒残差模块,EfficientVitLarge使用
#   3x3Conv + 1x1PWConv
#-----------------------------------------#
class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm_layer=(nn.BatchNorm2d, nn.BatchNorm2d),
        act_layer=(nn.ReLU6, None),
    ):
        super(FusedMBConv, self).__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        # 3x3Conv
        self.spatial_conv = ConvNormAct(
            in_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        # 1x1PWConv
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class LiteMLA(nn.Module):
    """Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm_layer=(None, nn.BatchNorm2d),
        act_layer=(None, None),
        kernel_func=nn.ReLU,
        scales=(5,),
        eps=1e-5,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)

        self.dim = dim
        self.qkv = ConvNormAct(
            in_channels,
            3 * total_dim,
            1,
            bias=use_bias[0],
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
        )
        # 通过多个大卷积核获取不同尺度特征
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 * total_dim,
                    3 * total_dim,
                    scale,
                    padding=get_same_padding(scale),
                    groups=3 * total_dim,
                    bias=use_bias[0],
                ),
                nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
            )
            for scale in scales
        ])
        self.kernel_func = kernel_func(inplace=False)

        self.proj = ConvNormAct(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            bias=use_bias[1],
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
        )

    def _attn(self, q, k, v):
        dtype = v.dtype
        q, k, v = q.float(), k.float(), v.float()
        kv = k.transpose(-1, -2) @ v    # [B, 8, 196, 16] -> [B, 8, 16, 196] @ [B, 8, 196, 17] = [B, 8, 16, 17]
        out = q @ kv                    # [B, 8, 196, 16] @ [B, 8, 16, 17] = [B, 8, 196, 17]
        out = out[..., :-1] / (out[..., -1:] + self.eps)    # [B, 8, 196, 17] get [B, 8, 196, 16]
        return out.to(dtype)

    def forward(self, x):
        B, _, H, W = x.shape

        # generate multi-scale q, k, v
        qkv = self.qkv(x)                   # [B, 64, 14, 14] -> [B, 3*64, 14, 14]
        multi_scale_qkv = [qkv]             # 多个大小相同的qkv(不同尺度)
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv)) # 通过多个大卷积核获取不同尺度特征
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)                                     # 拼接多尺度的qkv   2 * [B, 3*64, 14, 14] -> [B, 6*64, 14, 14]
        multi_scale_qkv = multi_scale_qkv.reshape(B, -1, 3 * self.dim, H * W).transpose(-1, -2) # [B, 6*64, 14, 14] -> [B, 8, 48, 196] -> [B, 8, 196, 48]
        q, k, v = multi_scale_qkv.chunk(3, dim=-1)                                              # [B, 8, 196, 48] -> 3 * [B, 8, 196, 16]

        # lightweight global attention
        q = self.kernel_func(q)                         # ReLU [B, 8, 196, 16] -> [B, 8, 196, 16]
        k = self.kernel_func(k)                         # ReLU [B, 8, 196, 16] -> [B, 8, 196, 16]
        v = F.pad(v, (0, 1), mode="constant", value=1.) # [B, 8, 196, 16] -> [B, 8, 196, 17]

        if not torch.jit.is_scripting():
            with torch.autocast(device_type=v.device.type, enabled=False):
                out = self._attn(q, k, v)
        else:
            out = self._attn(q, k, v)   # [B, 8, 196, 17] -> [B, 8, 196, 16]

        # final projection
        out = out.transpose(-1, -2).reshape(B, -1, H, W)    # [B, 8, 196, 16] -> [B, 8, 16, 196] -> [B, 128, 14, 14]
        out = self.proj(out)                                # [B, 128, 14, 14] -> [B, 64, 14, 14]
        return out


register_notrace_module(LiteMLA)


#----------------------#
#   LiteMSA + MBConv
#----------------------#
class EfficientVitBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        heads_ratio=1.0,
        head_dim=32,
        expand_ratio=4,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
    ):
        super(EfficientVitBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=head_dim,
                norm_layer=(None, norm_layer),
            ),
            nn.Identity(),
        )
        self.local_module = ResidualBlock(
            MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            ),
            nn.Identity(),
        )

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module] = None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()
        self.pre_norm = pre_norm if pre_norm is not None else nn.Identity()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x):
        res = self.main(self.pre_norm(x))
        if self.shortcut is not None:
            res = res + self.shortcut(x)
        return res


def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm_layer: str,
        act_layer: str,
        fewer_norm: bool = False,
        block_type: str = "default",
):
    assert block_type in ["default", "large", "fused"]
    if expand_ratio == 1:
        if block_type == "default":
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
        else:
            # Large
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
    else:
        if block_type == "default":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm_layer=(None, None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, act_layer, None),
            )
        else:
            # Large
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
    return block


#-------------------------------------------------------------------#
#   Conv + DSConv(3x3DWConv+1x1PWConv)/ConvBlock(3x3Conv+3x3Conv)
#-------------------------------------------------------------------#
class Stem(nn.Sequential):
    def __init__(self, in_chs, out_chs, depth, norm_layer, act_layer, block_type='default'):
        super().__init__()
        self.stride = 2

        self.add_module(
            'in_conv',
            ConvNormAct(
                in_chs, out_chs,
                kernel_size=3, stride=2, norm_layer=norm_layer, act_layer=act_layer,
            )
        )
        stem_block = 0
        for _ in range(depth):
            self.add_module(f'res{stem_block}', ResidualBlock(
                build_local_block(
                    in_channels=out_chs,
                    out_channels=out_chs,
                    stride=1,
                    expand_ratio=1,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    block_type=block_type,
                ),
                nn.Identity(),
            ))
            stem_block += 1


#------------------------------------#
#   4个stage,每个stage使用1次
#   stage1,2 use ResidualBlock
#   stage3,4 use EfficientVitBlock
#------------------------------------#
class EfficientVitStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            depth,
            norm_layer,
            act_layer,
            expand_ratio,
            head_dim,
            vit_stage=False,
    ):
        super(EfficientVitStage, self).__init__()
        # 每个stage开始都有一个 BMConv 用来下采样
        blocks = [ResidualBlock(
            build_local_block(
                in_channels=in_chs,
                out_channels=out_chs,
                stride=2,   # 下采样
                expand_ratio=expand_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                fewer_norm=vit_stage,
            ),
            None,
        )]
        in_chs = out_chs

        if vit_stage:
            # for stage 3, 4
            for _ in range(depth):
                blocks.append(
                    EfficientVitBlock(
                        in_channels=in_chs,
                        head_dim=head_dim,
                        expand_ratio=expand_ratio,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                )
        else:
            # for stage 1, 2,下标从1开始,总数包含开始的BMConv
            for i in range(1, depth):
                blocks.append(ResidualBlock(    # DSConv(3x3DWConv+1x1PWConv)
                    build_local_block(
                        in_channels=in_chs,
                        out_channels=out_chs,
                        stride=1,
                        expand_ratio=expand_ratio,
                        norm_layer=norm_layer,
                        act_layer=act_layer
                    ),
                    nn.Identity(),
                ))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


#------------------------------------#
#   4个stage,每个stage使用1次
#   stage1,2,3 use ResidualBlock
#   stage4 use EfficientVitBlock
#------------------------------------#
class EfficientVitLargeStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            depth,
            norm_layer,
            act_layer,
            head_dim,
            vit_stage=False,
            fewer_norm=False,
    ):
        super(EfficientVitLargeStage, self).__init__()
        blocks = [ResidualBlock(
            build_local_block(
                in_channels=in_chs,
                out_channels=out_chs,
                stride=2,
                expand_ratio=24 if vit_stage else 16,
                norm_layer=norm_layer,
                act_layer=act_layer,
                fewer_norm=vit_stage or fewer_norm,
                block_type='default' if fewer_norm else 'fused',
            ),
            None,
        )]
        in_chs = out_chs

        if vit_stage:
            # for stage 4
            for _ in range(depth):
                blocks.append(
                    EfficientVitBlock(
                        in_channels=in_chs,
                        head_dim=head_dim,
                        expand_ratio=6,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                )
        else:
            # for stage 1, 2, 3
            for i in range(depth):
                blocks.append(ResidualBlock(    # ConvBlock(3x3Conv+3x3Conv)
                    build_local_block(
                        in_channels=in_chs,
                        out_channels=out_chs,
                        stride=1,
                        expand_ratio=4,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        fewer_norm=fewer_norm,
                        block_type='default' if fewer_norm else 'fused',
                    ),
                    nn.Identity(),
                ))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_channels,
        widths,
        n_classes=1000,
        dropout=0.,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        global_pool='avg',
        norm_eps=1e-5,
    ):
        super(ClassifierHead, self).__init__()
        self.in_conv = ConvNormAct(in_channels, widths[0], 1, norm_layer=norm_layer, act_layer=act_layer)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True, input_fmt='NCHW')
        self.classifier = nn.Sequential(
            nn.Linear(widths[0], widths[1], bias=False),
            nn.LayerNorm(widths[1], eps=norm_eps),
            act_layer(inplace=True) if act_layer is not None else nn.Identity(),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(widths[1], n_classes, bias=True),
        )

    def forward(self, x, pre_logits: bool = False):
        x = self.in_conv(x)     # [B, 128, 7, 7] -> [B, 1024, 7, 7]
        x = self.global_pool(x) # [B, 1024, 7, 7] -> [B, 1024]
        if pre_logits:
            return x
        x = self.classifier(x)  # [B, 1024] -> [B, num_classes]
        return x


class EfficientVit(nn.Module):
    def __init__(
        self,
        in_chans=3,
        widths=(),      # 每个stage的宽度,包含stem
        depths=(),      # 每个stage的深度,包含stem
        head_dim=32,
        expand_ratio=4,
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.Hardswish,
        global_pool='avg',
        head_widths=(),
        drop_rate=0.0,
        num_classes=1000,
    ):
        super(EfficientVit, self).__init__()
        self.grad_checkpointing = False
        self.global_pool = global_pool
        self.num_classes = num_classes

        # input stem: Conv + DSConv(3x3DWConv+1x1PWConv)
        self.stem = Stem(in_chans, widths[0], depths[0], norm_layer, act_layer)
        stride = self.stem.stride

        # stages
        self.feature_info = []
        self.stages = nn.Sequential()
        in_channels = widths[0]
        for i, (w, d) in enumerate(zip(widths[1:], depths[1:])):
            self.stages.append(EfficientVitStage(
                in_channels,
                w,
                depth=d,
                norm_layer=norm_layer,
                act_layer=act_layer,
                expand_ratio=expand_ratio,
                head_dim=head_dim,
                vit_stage=i >= 2,   # stage2,3使用vit
            ))
            stride *= 2
            in_channels = w
            self.feature_info += [dict(num_chs=in_channels, reduction=stride, module=f'stages.{i}')]

        self.num_features = in_channels
        self.head_widths = head_widths
        self.head_dropout = drop_rate
        if num_classes > 0:
            self.head = ClassifierHead(
                self.num_features,
                self.head_widths,
                n_classes=num_classes,
                dropout=self.head_dropout,
                global_pool=self.global_pool,
            )
        else:
            if self.global_pool == 'avg':
                self.head = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
            else:
                self.head = nn.Identity()

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        if num_classes > 0:
            self.head = ClassifierHead(
                self.num_features,
                self.head_widths,
                n_classes=num_classes,
                dropout=self.head_dropout,
                global_pool=self.global_pool,
            )
        else:
            if self.global_pool == 'avg':
                self.head = SelectAdaptivePool2d(pool_type=self.global_pool, flatten=True)
            else:
                self.head = nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)        # [B, 3, 224, 224] -> [B, 8, 112, 112]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)  # [B, 8, 112, 112] -> [B, 128, 7, 7]
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)  # [B, 128, 7, 7] -> [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)    # [B, 3, 224, 224] -> [B, 128, 7, 7]
        x = self.forward_head(x)        # [B, 128, 7, 7] -> [B, num_classes]
        return x


class EfficientVitLarge(nn.Module):
    def __init__(
        self,
        in_chans=3,
        widths=(),      # 每个stage的宽度,包含stem
        depths=(),      # 每个stage的深度,包含stem
        head_dim=32,
        norm_layer=nn.BatchNorm2d,
        act_layer=GELUTanh,
        global_pool='avg',
        head_widths=(),
        drop_rate=0.0,
        num_classes=1000,
        norm_eps=1e-7,
    ):
        super(EfficientVitLarge, self).__init__()
        self.grad_checkpointing = False
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.norm_eps = norm_eps
        norm_layer = partial(norm_layer, eps=self.norm_eps)

        # input stem: Conv + ConvBlock(3x3Conv+3x3Conv)
        self.stem = Stem(in_chans, widths[0], depths[0], norm_layer, act_layer, block_type='large')
        stride = self.stem.stride

        # stages
        self.feature_info = []
        self.stages = nn.Sequential()
        in_channels = widths[0]
        for i, (w, d) in enumerate(zip(widths[1:], depths[1:])):
            self.stages.append(EfficientVitLargeStage(
                in_channels,
                w,
                depth=d,
                norm_layer=norm_layer,
                act_layer=act_layer,
                head_dim=head_dim,
                vit_stage=i >= 3,   # stage3使用vit
                fewer_norm=i >= 2,  # stage0,1使用FusedMBConv,stage2使用MBConv
            ))
            stride *= 2
            in_channels = w
            self.feature_info += [dict(num_chs=in_channels, reduction=stride, module=f'stages.{i}')]

        self.num_features = in_channels
        self.head_widths = head_widths
        self.head_dropout = drop_rate
        if num_classes > 0:
            self.head = ClassifierHead(
                self.num_features,
                self.head_widths,
                n_classes=num_classes,
                dropout=self.head_dropout,
                global_pool=self.global_pool,
                act_layer=act_layer,
                norm_eps=self.norm_eps,
            )
        else:
            if self.global_pool == 'avg':
                self.head = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
            else:
                self.head = nn.Identity()

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        if num_classes > 0:
            self.head = ClassifierHead(
                self.num_features,
                self.head_widths,
                n_classes=num_classes,
                dropout=self.head_dropout,
                global_pool=self.global_pool,
                norm_eps=self.norm_eps
            )
        else:
            if self.global_pool == 'avg':
                self.head = SelectAdaptivePool2d(pool_type=self.global_pool, flatten=True)
            else:
                self.head = nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)        # [B, 3, 224, 224] -> [B, 32, 112, 112]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)  # [B, 32, 112, 112] -> [B, 512, 7, 7]
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits) if pre_logits else self.head(x)  # [B, 512, 7, 7] -> [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)    # [B, 3, 224, 224] -> [B, 512, 7, 7]
        x = self.forward_head(x)        # [B, 512, 7, 7] -> [B, num_classes]
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.in_conv.conv',
        'classifier': 'head.classifier.4',
        'crop_pct': 0.95,
        'input_size': (3, 224, 224),
        'pool_size': (7, 7),
        **kwargs,
    }


default_cfgs = generate_default_cfgs({
    'efficientvit_b0.r224_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvit_b1.r224_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvit_b1.r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0,
    ),
    'efficientvit_b1.r288_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0,
    ),
    'efficientvit_b2.r224_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvit_b2.r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0,
    ),
    'efficientvit_b2.r288_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0,
    ),
    'efficientvit_b3.r224_in1k': _cfg(
        hf_hub_id='timm/',
    ),
    'efficientvit_b3.r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0,
    ),
    'efficientvit_b3.r288_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0,
    ),
    'efficientvit_l1.r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=1.0,
    ),
    'efficientvit_l2.r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=1.0,
    ),
    'efficientvit_l2.r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0,
    ),
    'efficientvit_l2.r288_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 288, 288), pool_size=(9, 9), crop_pct=1.0,
    ),
    'efficientvit_l2.r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    'efficientvit_l3.r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=1.0,
    ),
    'efficientvit_l3.r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=1.0,
    ),
    'efficientvit_l3.r320_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=1.0,
    ),
    'efficientvit_l3.r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0,
    ),
    # 'efficientvit_l0_sam.sam': _cfg(
    #     # hf_hub_id='timm/',
    #     input_size=(3, 512, 512), crop_pct=1.0,
    #     num_classes=0,
    # ),
    # 'efficientvit_l1_sam.sam': _cfg(
    #     # hf_hub_id='timm/',
    #     input_size=(3, 512, 512), crop_pct=1.0,
    #     num_classes=0,
    # ),
    # 'efficientvit_l2_sam.sam': _cfg(
    #     # hf_hub_id='timm/',f
    #     input_size=(3, 512, 512), crop_pct=1.0,
    #     num_classes=0,
    # ),
})


def _create_efficientvit(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    model = build_model_with_cfg(
        EfficientVit,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs
    )
    return model


def _create_efficientvit_large(variant, pretrained=False, **kwargs):
    out_indices = kwargs.pop('out_indices', (0, 1, 2, 3))
    model = build_model_with_cfg(
        EfficientVitLarge,
        variant,
        pretrained,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs
    )
    return model


def efficientvit_b0(pretrained=False, **kwargs):
    model_args = dict(  # 5 stages,包含stem
        widths=(8, 16, 32, 64, 128), depths=(1, 2, 2, 2, 2), head_dim=16, head_widths=(1024, 1280))
    return _create_efficientvit('efficientvit_b0', pretrained=pretrained, **dict(model_args, **kwargs))


def efficientvit_b1(pretrained=False, **kwargs):
    model_args = dict(
        widths=(16, 32, 64, 128, 256), depths=(1, 2, 3, 3, 4), head_dim=16, head_widths=(1536, 1600))
    return _create_efficientvit('efficientvit_b1', pretrained=pretrained, **dict(model_args, **kwargs))


def efficientvit_b2(pretrained=False, **kwargs):
    model_args = dict(
        widths=(24, 48, 96, 192, 384), depths=(1, 3, 4, 4, 6), head_dim=32, head_widths=(2304, 2560))
    return _create_efficientvit('efficientvit_b2', pretrained=pretrained, **dict(model_args, **kwargs))


def efficientvit_b3(pretrained=False, **kwargs):
    model_args = dict(
        widths=(32, 64, 128, 256, 512), depths=(1, 4, 6, 6, 9), head_dim=32, head_widths=(2304, 2560))
    return _create_efficientvit('efficientvit_b3', pretrained=pretrained, **dict(model_args, **kwargs))


def efficientvit_l1(pretrained=False, **kwargs):
    model_args = dict(
        widths=(32, 64, 128, 256, 512), depths=(1, 1, 1, 6, 6), head_dim=32, head_widths=(3072, 3200))
    return _create_efficientvit_large('efficientvit_l1', pretrained=pretrained, **dict(model_args, **kwargs))


def efficientvit_l2(pretrained=False, **kwargs):
    model_args = dict(
        widths=(32, 64, 128, 256, 512), depths=(1, 2, 2, 8, 8), head_dim=32, head_widths=(3072, 3200))
    return _create_efficientvit_large('efficientvit_l2', pretrained=pretrained, **dict(model_args, **kwargs))


def efficientvit_l3(pretrained=False, **kwargs):
    model_args = dict(
        widths=(64, 128, 256, 512, 1024), depths=(1, 2, 2, 8, 8), head_dim=32, head_widths=(6144, 6400))
    return _create_efficientvit_large('efficientvit_l3', pretrained=pretrained, **dict(model_args, **kwargs))


# FIXME will wait for v2 SAM models which are pending
# def efficientvit_l0_sam(pretrained=False, **kwargs):
#     # only backbone for segment-anything-model weights
#     model_args = dict(
#         widths=(32, 64, 128, 256, 512), depths=(1, 1, 1, 4, 4), head_dim=32, num_classes=0, norm_eps=1e-6)
#     return _create_efficientvit_large('efficientvit_l0_sam', pretrained=pretrained, **dict(model_args, **kwargs))
#
#
# def efficientvit_l1_sam(pretrained=False, **kwargs):
#     # only backbone for segment-anything-model weights
#     model_args = dict(
#         widths=(32, 64, 128, 256, 512), depths=(1, 1, 1, 6, 6), head_dim=32, num_classes=0, norm_eps=1e-6)
#     return _create_efficientvit_large('efficientvit_l1_sam', pretrained=pretrained, **dict(model_args, **kwargs))
#
#
# def efficientvit_l2_sam(pretrained=False, **kwargs):
#     # only backbone for segment-anything-model weights
#     model_args = dict(
#         widths=(32, 64, 128, 256, 512), depths=(1, 2, 2, 8, 8), head_dim=32, num_classes=0, norm_eps=1e-6)
#     return _create_efficientvit_large('efficientvit_l2_sam', pretrained=pretrained, **dict(model_args, **kwargs))


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = efficientvit_b0(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]

    if False:
        onnx_path = 'efficientvit_b0.onnx'
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
