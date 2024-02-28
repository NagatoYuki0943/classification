"""
An implementation of GhostNet & GhostNetV2 Models as defined in:
GhostNet: More Features from Cheap Operations. https://arxiv.org/abs/1911.11907
GhostNetV2: Enhance Cheap Operation with Long-Range Attention. https://proceedings.neurips.cc/paper_files/paper/2022/file/40b60852a4abdaa696b5a1a78da34635-Paper-Conference.pdf

The train script & code of models at:
Original model: https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch
Original model: https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnetv2_pytorch/model/ghostnetv2_torch.py
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, make_divisible, create_act_layer
from timm.models._builder import build_model_with_cfg
from timm.models._efficientnet_blocks import ConvBnAct
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs

__all__ = ['GhostNet']


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.ReLU,
            gate_layer=nn.Sigmoid, force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


_SE_LAYER = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=partial(make_divisible, divisor=4))


#----------------------------------------#
#               in
#                │
#           primary_conv(1x1Conv)
#                │
#      ┌─────────┤
#      │         │
#      │   cheap_operation(3x3DWConv)
#      │         │
#      └────────cat
#                │
#               out
#----------------------------------------#
class GhostModule(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=1,
            ratio=2,
            dw_size=3,
            stride=1,
            use_act=True,
            act_layer=nn.ReLU,
    ):
        super(GhostModule, self).__init__()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        #------------------------------------------------------#
        #   1x1Conv降低通道数,特征浓缩 通道数变为 oup_channel/2
        #   跨通道的特征提取
        #------------------------------------------------------#
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

        #------------------------------------------------------#
        #   3x3DWConv对降低的通道数进行计算
        #   跨特征点的特诊提取
        #------------------------------------------------------#
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_chs, new_chs, dw_size, 1, dw_size // 2, groups=init_chs, bias=False),
            nn.BatchNorm2d(new_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)           # [B, 16, 112, 112] -> [B, 8, 112, 112]
        x2 = self.cheap_operation(x1)       # [B, 8, 112, 112]  -> [B, 8, 112, 112]
        out = torch.cat([x1, x2], dim=1)    # [B, 8, 112, 112] cat [B, 8, 112, 112] = [B, 16, 112, 112]
        return out[:, :self.out_chs, :, :]  # [B, 16, 112, 112] get [B, 16, 112, 112]


#-------------------------------------------------------#
#                            in
#         GhostModule         │         DFC attention
#              ┌──────────────┴──────────────┐
#              │                             │
#              │                       2x2avg_pool2d
#              │                             │
#              │                          1x1Conv
#         primary_conv(1x1Conv)              │
#              │                             │
#    ┌─────────┤                          1x5Conv
#    │         │                             │
#    │   cheap_operation(3x3DWConv)          │
#    │         │                          5x1Conv
#    └────────cat                            │
#              │                             │
#              │                       F.interpolate
#              │                             │
#              │                          sigmoid
#              │                             │
#              └───────────── * ─────────────┘
#                             │
#                            out
#-------------------------------------------------------#
class GhostModuleV2(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=1,
            ratio=2,
            dw_size=3,
            stride=1,
            use_act=True,
            act_layer=nn.ReLU,
    ):
        super().__init__()
        self.gate_fn = nn.Sigmoid()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        #------------------------------------------------------#
        #   1x1Conv降低通道数,特征浓缩 通道数变为 oup_channel/2
        #   跨通道的特征提取
        #------------------------------------------------------#
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

        #------------------------------------------------------#
        #   3x3DWConv对降低的通道数进行计算
        #   跨特征点的特诊提取
        #------------------------------------------------------#
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_chs, new_chs, dw_size, 1, dw_size // 2, groups=init_chs, bias=False),
            nn.BatchNorm2d(new_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

        #------------------------------------------------------#
        #   DFC attention
        #   1x1Conv(stride = 1 or 2) -> 1x5Conv -> 5x1Conv
        #------------------------------------------------------#
        self.short_conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=out_chs, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.Conv2d(out_chs, out_chs, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_chs, bias=False),
            nn.BatchNorm2d(out_chs),
        )

    def forward(self, x):
        res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_chs, :, :] * F.interpolate(
            self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode='nearest')


#------------------------------------------------------------------------------#
#   倒残差结构,mid_chs增多
#   和MobileNet的不同点有2个PWConv使用了GhostModule,stride!=1时才使用中间的DWConv
#   v1 v2差别在于ghost1使用的GhostModule不同
#------------------------------------------------------------------------------#
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            act_layer=nn.ReLU,
            se_ratio=0.,
            mode='original',
    ):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        #------------------------------------------------------#
        #  Point-wise 提高通道数
        #------------------------------------------------------#
        if mode == 'original':
            self.ghost1 = GhostModule(in_chs, mid_chs, use_act=True, act_layer=act_layer)
        else:
            self.ghost1 = GhostModuleV2(in_chs, mid_chs, use_act=True, act_layer=act_layer)

        #------------------------------------------------------#
        #   Depth-wise convolution
        #   只有步长不为1才使用
        #------------------------------------------------------#
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size-1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        #------------------------------------------------------#
        #   Squeeze-and-excitation
        #------------------------------------------------------#
        self.se = _SE_LAYER(mid_chs, rd_ratio=se_ratio) if has_se else None

        #------------------------------------------------------#
        #  Point-wise 降低通道数
        #------------------------------------------------------#
        self.ghost2 = GhostModule(mid_chs, out_chs, use_act=False)

        #------------------------------------------------------#
        #   shortcut
        #   3x3DWConv -> 1x1Conv
        #------------------------------------------------------#
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        shortcut = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(shortcut)
        return x


class GhostNet(nn.Module):
    def __init__(
            self,
            cfgs,
            num_classes=1000,
            width=1.0,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.2,
            version='v1',
    ):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        assert output_stride == 32, 'only output_stride==32 is valid, dilation not supported'
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        #------------------------------------------------------#
        #   building first layer
        #   [B, 3, 224, 224] -> [B, 16, 112, 112]
        #------------------------------------------------------#
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_chans, stem_chs, 3, 2, 1, bias=False)
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=f'conv_stem'))
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = nn.ReLU(inplace=True)
        prev_chs = stem_chs

        #------------------------------------------------------#
        #   building inverted residual blocks
        #   [B, 16, 112, 112] -> [B, 160, 7, 7]
        #------------------------------------------------------#
        stages = nn.ModuleList([])
        stage_idx = 0
        layer_idx = 0
        net_stride = 2
        for cfg in self.cfgs:
            layers = []
            s = 1
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layer_kwargs = {}
                if version == 'v2' and layer_idx > 1:
                    layer_kwargs['mode'] = 'attn'
                layers.append(GhostBottleneck(prev_chs, mid_chs, out_chs, k, s, se_ratio=se_ratio, **layer_kwargs))
                prev_chs = out_chs
                layer_idx += 1
            if s > 1:
                net_stride *= 2
                self.feature_info.append(dict(
                    num_chs=prev_chs, reduction=net_stride, module=f'blocks.{stage_idx}'))
            stages.append(nn.Sequential(*layers))
            stage_idx += 1

        #------------------------------------------------------#
        #   [B, 160, 7, 7] -> [B, 960, 7, 7]
        #------------------------------------------------------#
        out_chs = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(prev_chs, out_chs, 1)))
        self.pool_dim = prev_chs = out_chs

        self.blocks = nn.Sequential(*stages)

        #------------------------------------------------------#
        #   分类层
        #   [B, 960, 7, 7] -> [B, num_classes]
        #------------------------------------------------------#
        self.num_features = out_chs = 1280
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv2d(prev_chs, out_chs, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(out_chs, num_classes) if num_classes > 0 else nn.Identity()

        # FIXME init

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.pool_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)   # [B, 3, 224, 224] -> [B, 16, 112, 112]
        x = self.bn1(x)
        x = self.act1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)  # [B, 16, 112, 112] -> [B, 960, 7, 7]
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)    # [B, 3, 224, 224] -> [B, 960, 7, 7]
        x = self.forward_head(x)        # [B, 960, 7, 7] -> [B, num_classes]
        return x


def checkpoint_filter_fn(state_dict, model: nn.Module):
    out_dict = {}
    for k, v in state_dict.items():
        if 'total' in k:
            continue
        out_dict[k] = v
    return out_dict


def _create_ghostnet(variant, width=1.0, pretrained=False, **kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # k: 卷积核大小,表示跨特征点能力
        # t: GhostBottleneck的 mid_chs
        # c: GhostBottleneck的 out_chs
        # SE:是否使用注意力机制,0不使用
        # s: 步长
        # stage1    114, 114, 16 -> 114, 114, 16
        [[3,  16,  16, 0, 1]],
        # stage2    114, 114, 16 -> 52, 52, 24
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3    52, 52, 24 -> 28, 28, 40
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4    28, 28, 40 -> 14, 14, 80 -> 14, 14, 112
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5    14, 14, 112 -> 7, 7, 160
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        **kwargs,
    )
    return build_model_with_cfg(
        GhostNet,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'ghostnet_050.untrained': _cfg(),
    'ghostnet_100.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth'
    ),
    'ghostnet_130.untrained': _cfg(),
    'ghostnetv2_100.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_10.pth.tar'
    ),
    'ghostnetv2_130.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_13.pth.tar'
    ),
    'ghostnetv2_160.in1k': _cfg(
        hf_hub_id='timm/',
        # url='https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/GhostNetV2/ck_ghostnetv2_16.pth.tar'
    ),
})


def ghostnet_050(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNet-0.5x """
    model = _create_ghostnet('ghostnet_050', width=0.5, pretrained=pretrained, **kwargs)
    return model


def ghostnet_100(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNet-1.0x """
    model = _create_ghostnet('ghostnet_100', width=1.0, pretrained=pretrained, **kwargs)
    return model


def ghostnet_130(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNet-1.3x """
    model = _create_ghostnet('ghostnet_130', width=1.3, pretrained=pretrained, **kwargs)
    return model


def ghostnetv2_100(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNetV2-1.0x """
    model = _create_ghostnet('ghostnetv2_100', width=1.0, pretrained=pretrained, version='v2', **kwargs)
    return model


def ghostnetv2_130(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNetV2-1.3x """
    model = _create_ghostnet('ghostnetv2_130', width=1.3, pretrained=pretrained, version='v2', **kwargs)
    return model


def ghostnetv2_160(pretrained=False, **kwargs) -> GhostNet:
    """ GhostNetV2-1.6x """
    model = _create_ghostnet('ghostnetv2_160', width=1.6, pretrained=pretrained, version='v2', **kwargs)
    return model


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = ghostnet_100(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]

    # 查看结构
    if False:
        onnx_path = 'ghostnet_100.onnx'
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
