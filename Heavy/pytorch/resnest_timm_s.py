""" ResNeSt Models

Paper: `ResNeSt: Split-Attention Networks` - https://arxiv.org/abs/2004.08955

Adapted from original PyTorch impl w/ weights at https://github.com/zhanghang1989/ResNeSt by Hang Zhang

Modified for torchscript compat, and consistency with timm by Ross Wightman
"""
import torch
from torch import nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SplitAttn
from timm.layers.helpers import make_divisible
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model, generate_default_cfgs
from timm.models.resnet import ResNet


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)   # [B, 128, 1, 1]
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2) # [B, 128, 1, 1] -> [B, 1, 2, 64] -> [B, 2, 1, 64]
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttn(nn.Module):
    """Split-Attention (aka Splat)
    """
    def __init__(
            self,
            in_channels,
            out_channels=None,
            kernel_size=3,
            stride=1,
            padding=None,
            dilation=1,
            groups=1,
            bias=False,
            radix=2,
            rd_ratio=0.25,
            rd_channels=None,
            rd_divisor=8,
            act_layer=nn.ReLU,
            norm_layer=None,
            drop_layer=None,
            **kwargs
    ):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        mid_chs = out_channels * radix
        if rd_channels is None:
            attn_chs = make_divisible(in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)    # [B, 64, 56, 56] -> [B, 128, 56, 56]
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)

        B, RC, H, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))  # [B, 128, 56, 56] -> [B, 2, 64, 56, 56]
            x_gap = x.sum(dim=1)                                    # [B, 2, 64, 56, 56] sum [B, 64, 56, 56]
        else:
            x_gap = x
        x_gap = x_gap.mean((2, 3), keepdim=True)                    # [B, 64, 56, 56] -> [B, 64, 1, 1]
        x_gap = self.fc1(x_gap)                                     # [B, 64, 1, 1] -> [B, 32, 1, 1]
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)                                    # [B, 32, 1, 1] -> [B, 128, 1, 1]

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)            # [B, 128, 1, 1] -> [B, 2, 1, 64] 2代表2个radix
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)  # [B, 2, 64, 56, 56] * ([B, 2, 1, 64] -> [B, 2, 64, 1, 1]) = [B, 2, 64, 56, 56] -> [B, 64, 56, 56]
        else:
            out = x * x_attn
        return out.contiguous()


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            radix=1,
            cardinality=1,  # 分组卷积分组数
            base_width=64,
            avd=False,
            avd_first=False,
            is_first=False,
            reduce_first=1,
            dilation=1,
            first_dilation=None,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            attn_layer=None,
            aa_layer=None,
            drop_block=None,
            drop_path=None,
    ):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        assert attn_layer is None  # not supported
        assert aa_layer is None  # TODO not yet supported
        assert drop_path is None  # TODO not yet supported

        group_width = int(planes * (base_width / 64.)) * cardinality
        first_dilation = first_dilation or dilation
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0
        self.radix = radix

        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.act1 = act_layer(inplace=True)
        self.avd_first = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and avd_first else None

        if self.radix >= 1:
            self.conv2 = SplitAttn(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer, drop_layer=drop_block)
            self.bn2 = nn.Identity()
            self.drop_block = nn.Identity()
            self.act2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.drop_block = drop_block() if drop_block is not None else nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.avd_last = nn.AvgPool2d(3, avd_stride, padding=1) if avd_stride > 0 and not avd_first else None

        self.conv3 = nn.Conv2d(group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)
        self.act3 = act_layer(inplace=True)
        self.downsample = downsample

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        shortcut = x            # [B, 64, 56, 56]

        out = self.conv1(x)     # [B, 64, 56, 56] -> [B, 64, 56, 56]
        out = self.bn1(out)
        out = self.act1(out)

        if self.avd_first is not None:
            out = self.avd_first(out)

        # SplitAttn
        out = self.conv2(out)   # [B, 64, 56, 56] -> [B, 64, 56, 56]
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.conv3(out)   # [B, 64, 56, 56] -> [B, 256, 56, 56]
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)   # [B, 64, 56, 56] -> [B, 256, 56, 56]

        out += shortcut         # [B, 256, 56, 56] + [B, 256, 56, 56] = [B, 256, 56, 56]
        out = self.act3(out)
        return out


def _create_resnest(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet,
        variant,
        pretrained,
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1.0', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'resnest14d.gluon_in1k': _cfg(hf_hub_id='timm/'),
    'resnest26d.gluon_in1k': _cfg(hf_hub_id='timm/'),
    'resnest50d.in1k': _cfg(hf_hub_id='timm/'),
    'resnest101e.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8)),
    'resnest200e.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 320, 320), pool_size=(10, 10), crop_pct=0.909, interpolation='bicubic'),
    'resnest269e.in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 416, 416), pool_size=(13, 13), crop_pct=0.928, interpolation='bicubic'),
    'resnest50d_4s2x40d.in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic'),
    'resnest50d_1s4x24d.in1k': _cfg(
        hf_hub_id='timm/',
        interpolation='bicubic')
})


def resnest14d(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-14d model. Weights ported from GluonCV.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[1, 1, 1, 1],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest14d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


def resnest26d(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-26d model. Weights ported from GluonCV.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[2, 2, 2, 2],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest26d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


def resnest50d(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-50d model. Matches paper ResNeSt-50 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'd' for deep stem, stem_width 32, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest50d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


def resnest101e(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-101e model. Matches paper ResNeSt-101 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 23, 3],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest101e', pretrained=pretrained, **dict(model_kwargs, **kwargs))


def resnest200e(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-200e model. Matches paper ResNeSt-200 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 24, 36, 3],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest200e', pretrained=pretrained, **dict(model_kwargs, **kwargs))


def resnest269e(pretrained=False, **kwargs) -> ResNet:
    """ ResNeSt-269e model. Matches paper ResNeSt-269 model, https://arxiv.org/abs/2004.08955
    Since this codebase supports all possible variations, 'e' for deep stem, stem_width 64, avg in downsample.
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 30, 48, 8],
        stem_type='deep', stem_width=64, avg_down=True, base_width=64, cardinality=1,
        block_args=dict(radix=2, avd=True, avd_first=False))
    return _create_resnest('resnest269e', pretrained=pretrained, **dict(model_kwargs, **kwargs))


def resnest50d_4s2x40d(pretrained=False, **kwargs) -> ResNet:
    """ResNeSt-50 4s2x40d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=40, cardinality=2,
        block_args=dict(radix=4, avd=True, avd_first=True))
    return _create_resnest('resnest50d_4s2x40d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


def resnest50d_1s4x24d(pretrained=False, **kwargs) -> ResNet:
    """ResNeSt-50 1s4x24d from https://github.com/zhanghang1989/ResNeSt/blob/master/ablation.md
    """
    model_kwargs = dict(
        block=ResNestBottleneck, layers=[3, 4, 6, 3],
        stem_type='deep', stem_width=32, avg_down=True, base_width=24, cardinality=4,
        block_args=dict(radix=1, avd=True, avd_first=True))
    return _create_resnest('resnest50d_1s4x24d', pretrained=pretrained, **dict(model_kwargs, **kwargs))


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = resnest14d(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]

    # 查看结构
    if False:
        onnx_path = 'resnest14d.onnx'
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
