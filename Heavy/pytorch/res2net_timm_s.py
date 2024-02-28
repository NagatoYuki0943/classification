""" Res2Net and Res2NeXt
Adapted from Official Pytorch impl at: https://github.com/gasvn/Res2Net/
Paper: `Res2Net: A New Multi-scale Backbone Architecture` - https://arxiv.org/abs/1904.01169
"""
import math

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model, generate_default_cfgs
from timm.models.resnet import ResNet

__all__ = []


#------------------------------------------------------#
#   stype='normal' 通道宽高不变
#                           in
#      ┌─────────────────────┤
#      │                     │
#      │                 conv1(1x1)
#      │                     │
#      │            ┌──────split─────────────────┐
#      │            │        │ └───────┐         │
#      │            │        │         │         │
#      │            │   conv2_1(3x3)   │         │
#      │            │        ├────────add        │
#      |            |        |         |         │
#      │            │        │    conv2_2(3x3)   │
#      |            |        │         │         │
#      │            │        │         ├────────add
#      |            |        |         |         |
#      │            │        │         │    conv2_3(3x3)
#      │            │        │         │         │
#      │            │        │┌────────┘         │
#      │            └─────concat─────────────────┘
#      │                     │
#      │                 conv3(1X1)
#      │                     │
#      └───────────────────-add
#                            │
#                           out
#
#   stype='stage' 通道变化或者下采样或者两者都有
#   stage1的第1个block只变化通道,并不影响中间通道的相加,但是没有相加
#                           in
#      ┌─────────────────────┤
#      │                     │
#      │                 conv1(1x1)
#      │                     │
#      │            ┌──────split─────────────────┐
#      │            │        │ └───────┐         │
#      │            │        │         │         │
#      │            │   conv2_1(3x3)   │         │
#      │            │        │         │         │
#  downsample    avg_pool    |         |         │
#      │            │        │    conv2_2(3x3)   │
#      |            |        │         │         │
#      │            │        │         │         │
#      │            │        │         │    conv2_3(3x3)
#      │            │        │┌────────┘         │
#      │            └─────concat─────────────────┘
#      │                     │
#      │                 conv3(1X1)
#      │                     │
#      └───────────────────-add
#                            │
#                           out
#------------------------------------------------------#
class Bottle2neck(nn.Module):
    """ Res2Net/Res2NeXT Bottleneck
    Adapted from https://github.com/gasvn/Res2Net/blob/master/res2net.py
    """
    expansion = 4   # out_channel扩展倍率

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            cardinality=1,
            base_width=26,
            scale=4,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            attn_layer=None,
            **_,
    ):
        super(Bottle2neck, self).__init__()
        self.scale = scale
        self.is_first = stride > 1 or downsample is not None
        self.num_scales = max(1, scale - 1)
        # 基础宽度
        width = int(math.floor(planes * (base_width / 64.0))) * cardinality
        self.width = width
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        #-------------------------------#
        #   conv1
        #   [B, 64, 56, 56] -> [B, 104, 56, 56]
        #-------------------------------#
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width * scale)

        #-------------------------------#
        #   convs
        #   中间3个分支的创建
        #   个数 = scale - 1,有一个分支不计算
        #-------------------------------#
        convs = []
        bns = []
        for i in range(self.num_scales):
            convs.append(nn.Conv2d(
                width, width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False))
            bns.append(norm_layer(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        #--------------------------------------#
        #   下采样层
        #   stage1的下采样步长为1,宽高不变
        #--------------------------------------#
        if self.is_first:
            # FIXME this should probably have count_include_pad=False, but hurts original weights
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.pool = None

        #-------------------------------#
        #   conv3
        #   [B, 104, 56, 56] -> [B, 256, 56, 56]
        #-------------------------------#
        self.conv3 = nn.Conv2d(width * scale, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.se = attn_layer(outplanes) if attn_layer is not None else None

        self.relu = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x

        # [B, 64, 56, 56] -> [B, 104, 56, 56]   stage1第1次
        # [B, 256, 56, 56]-> [B, 208, 56, 56]   stage2第1次
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # [B, 104, 56, 56] split [B, 26, 56, 56] * 4    stage1第1次
        # [B, 208, 56, 56] split [B, 52, 56, 56] * 4    stage2第1次
        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]  # redundant, for torchscript
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                # 第1步或者有下采样层直接赋值
                # 有下采样层不相加
                sp = spx[i]
            else:
                # 下面要相加
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:  # self.is_first == True, None check for torchscript
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])

        # 拼接不计算的部分,要进行池化(池化的stride可能为1,所以宽高会不变)
        # [B,  78, 56, 56] cat pool([B, 26, 56, 56]) = [B, 104, 56, 56]   pool s=1  stage1第1次
        # [B, 156, 28, 28] cat pool([B, 52, 56, 56]) = [B, 208, 28, 28]   pool s=2  stage2第1次
        out = torch.cat(spo, 1)

        # [B, 104, 56, 56] -> [B, 256, 56, 56]  stage1第1次
        # [B, 208, 28, 28] -> [B, 512, 28, 28]  stage2第1次
        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        # 有下采样是残差部分也要下采样
        if self.downsample is not None:
            # [B,  64, 56, 56] -> [B, 256, 56, 56]  stage1第1次 s=1
            # [B, 256, 56, 56] -> [B, 512, 28, 28]  stage2第1次 s=2
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out


def _create_res2net(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'res2net50_26w_4s.in1k': _cfg(hf_hub_id='timm/'),
    'res2net50_48w_2s.in1k': _cfg(hf_hub_id='timm/'),
    'res2net50_14w_8s.in1k': _cfg(hf_hub_id='timm/'),
    'res2net50_26w_6s.in1k': _cfg(hf_hub_id='timm/'),
    'res2net50_26w_8s.in1k': _cfg(hf_hub_id='timm/'),
    'res2net101_26w_4s.in1k': _cfg(hf_hub_id='timm/'),
    'res2next50.in1k': _cfg(hf_hub_id='timm/'),
    'res2net50d.in1k': _cfg(hf_hub_id='timm/', first_conv='conv1.0'),
    'res2net101d.in1k': _cfg(hf_hub_id='timm/', first_conv='conv1.0'),
})


def res2net50_26w_4s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Res2Net-50 26w4s model.
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=26, block_args=dict(scale=4))
    return _create_res2net('res2net50_26w_4s', pretrained, **dict(model_args, **kwargs))


def res2net101_26w_4s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Res2Net-101 26w4s model.
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 23, 3], base_width=26, block_args=dict(scale=4))
    return _create_res2net('res2net101_26w_4s', pretrained, **dict(model_args, **kwargs))


def res2net50_26w_6s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Res2Net-50 26w6s model.
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=26, block_args=dict(scale=6))
    return _create_res2net('res2net50_26w_6s', pretrained, **dict(model_args, **kwargs))


def res2net50_26w_8s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Res2Net-50 26w8s model.
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=26, block_args=dict(scale=8))
    return _create_res2net('res2net50_26w_8s', pretrained, **dict(model_args, **kwargs))


def res2net50_48w_2s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Res2Net-50 48w2s model.
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=48, block_args=dict(scale=2))
    return _create_res2net('res2net50_48w_2s', pretrained, **dict(model_args, **kwargs))


def res2net50_14w_8s(pretrained=False, **kwargs) -> ResNet:
    """Constructs a Res2Net-50 14w8s model.
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=14, block_args=dict(scale=8))
    return _create_res2net('res2net50_14w_8s', pretrained, **dict(model_args, **kwargs))


def res2next50(pretrained=False, **kwargs) -> ResNet:
    """Construct Res2NeXt-50 4s
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=4, cardinality=8, block_args=dict(scale=4))
    return _create_res2net('res2next50', pretrained, **dict(model_args, **kwargs))


def res2net50d(pretrained=False, **kwargs) -> ResNet:
    """Construct Res2Net-50
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 6, 3], base_width=26, stem_type='deep',
        avg_down=True, stem_width=32, block_args=dict(scale=4))
    return _create_res2net('res2net50d', pretrained, **dict(model_args, **kwargs))


def res2net101d(pretrained=False, **kwargs) -> ResNet:
    """Construct Res2Net-50
    """
    model_args = dict(
        block=Bottle2neck, layers=[3, 4, 23, 3], base_width=26, stem_type='deep',
        avg_down=True, stem_width=32, block_args=dict(scale=4))
    return _create_res2net('res2net101d', pretrained, **dict(model_args, **kwargs))


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = res2net50_26w_4s(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]

    # 查看结构
    if False:
        onnx_path = 'res2net50_26w_4s.onnx'
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
