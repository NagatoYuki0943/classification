# Modified from
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/anynet.py
# https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py
"""
pytorch源码

stem

body

head

400mf和mobilenet_v3准确率差不多,训练速度更快

192行
MobileNetV2 V3 EfficientNetV1 V2 都跳过了不变化维度的第一个1x1Conv
但是RegNet没跳过
这里使用 width_in == w_b 作为判断即可跳过,因为维度维度变化为 c_i => c_i => c_i 或者 c_{i-1} => c_i => c_i
主要就是第一个1x1Conv变化维度

y使用了se注意力机制,x没使用
"""

import math
import torch

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url


__all__ = [
    "RegNet",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_1_6gf",
    "regnet_y_3_2gf",
    "regnet_y_8gf",
    "regnet_y_16gf",
    "regnet_y_32gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_1_6gf",
    "regnet_x_3_2gf",
    "regnet_x_8gf",
    "regnet_x_16gf",
    "regnet_x_32gf",
]


model_urls = {
    "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
    "regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
    "regnet_y_1_6gf": "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
    "regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
    "regnet_y_8gf": "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
    "regnet_y_16gf": "https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
    "regnet_y_32gf": "https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
    "regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
    "regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
    "regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
    "regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
    "regnet_x_8gf": "https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
    "regnet_x_16gf": "https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
    "regnet_x_32gf": "https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    将卷积核个数(输出通道个数)调整为最接近round_nearest的整数倍,就是8的整数倍,对硬件更加友好
    v:          输出通道个数
    divisor:    奇数,必须将ch调整为它的整数倍
    min_value:  最小通道数

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(nn.Sequential):
    """
    标准卷积块
    Conv + BN + ReLU
    默认宽高不变
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[
            Callable[..., nn.Module]
        ] = nn.BatchNorm2d,  # BN,默认BatchNorm2d
        activation_layer: Optional[
            Callable[..., nn.Module]
        ] = nn.ReLU,  # 激活函数,默认ReLU
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        # 自动调整padding,让宽高不变
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=norm_layer is None,
            )
        ]
        # 默认BN
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        # 默认ReLU
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


class SqueezeExcitation(nn.Module):
    """
    注意力机制
        对特征矩阵每一个channel进行池化,得到长度为channel的一维向量,使用两个全连接层,
        两个线性层的长度,最后得到权重,然后乘以每层矩阵的原值
        线性层长度变化: expand_c -> input_c / 4 -> expand_c

    fc1的输出是该Block的in_channel的四分之一
    """

    def __init__(
        self,
        input_channels: int,  # in_channels&out_channels
        squeeze_channels: int,  # 中间维度,是输入block维度的1/4
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 两个卷积作为全连接层,kernel为1
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        # [batch, channel, height, width] -> [batch, channel, 1, 1]
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)

        # [batch, channel, 1, 1] * [batch, channel, height, width]
        # 高维度矩阵相乘是最后两个维度相乘,所以是 [1, 1] 点乘 [h, w]
        return scale * input


class SimpleStemIN(ConvNormActivation):
    """
    开始的stem
    [b, 3, h, w] => [b, 32, h/2, w/2]
    Simple stem for ImageNet: 3x3, BN, ReLU.
    """

    def __init__(
        self,
        width_in: int,  # in_channels
        width_out: int,  # out_channels
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        # [b, 3, h, w] => [b, 32, h/2, w/2]
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )


class BottleneckTransform(nn.Sequential):
    """
    block中的3层卷积+SE
    Bottleneck transformation: 1x1, 3x3 [+SE], 1x1.
    最后的1x1Conv没有激活函数
    """

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,  # 每一个group的channels
        bottleneck_multiplier: float,  # 第一个1x1Conv维度变化倍率,默认为1,代表输出维度不变化,这样效果最好
        se_ratio: Optional[float],  # SE机制倍率 0.25
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()

        # 确定中间维度变化,默认width_out=1,维度不变化
        w_b = int(round(width_out * bottleneck_multiplier))

        # 分组数量 = in_channels / group_width
        g = w_b // group_width

        # 1x1
        # MobileNetV2 V3 EfficientNetV1 V2 都跳过了不变化维度的第一个1x1Conv
        # 这里使用 width_in == w_b 作为判断即可跳过,因为维度维度变化为 c_i => c_i => c_i 或者 c_i-1 => c_i => c_i,主要就是第一个1x1Conv变化维度
        layers["a"] = ConvNormActivation(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        # 3x3
        layers["b"] = ConvNormActivation(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            groups=g,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # 注意力机制
        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # 中间维度,是输入block维度的1/4
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,  # in_channels&out_channels
                squeeze_channels=width_se_out,  # 中间维度,是输入block维度的1/4
                activation=activation_layer,
            )

        # 1x1 不要激活函数
        layers["c"] = ConvNormActivation(
            w_b,
            width_out,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        # layers是字典,不需要拆开
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """
    一共4个stage,这是每个stage的单个block
    Residual bottleneck block: x + F(x), F = bottleneck transform.
    """

    def __init__(
        self,
        width_in: int,  # in_channels
        width_out: int,  # out_channels
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,  # 每一个group的channels
        bottleneck_multiplier: float = 1.0,  # 第一个1x1Conv维度变化倍率,默认为1,代表输出维度不变化,这样效果最好
        se_ratio: Optional[float] = None,  # SE机制倍率 0.25
    ) -> None:
        super().__init__()

        # 短接层变化默认没有
        self.proj = None
        # 维度变化或者步长不为1要特殊处理
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        # 1x1 + 3x3 + [SE] + 1x1
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # 维度变化或者步长不为1要特殊处理
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """
    4个阶段
    添加block,名称: block1-0 block2-2
    AnyNet stage (sequence of blocks w/ the same output shape).
    """

    def __init__(
        self,
        width_in: int,  # stage的in_channels
        width_out: int,  # stage的out_channels,第一个block就转换
        stride: int,  # 步长,第一个block使用
        depth: int,  # 重复次数
        block_constructor: Callable[..., nn.Module],  # ResBottleneckBlock
        norm_layer: Callable[..., nn.Module],  # BN
        activation_layer: Callable[..., nn.Module],  # 激活函数
        group_width: int,  # 分组宽度
        bottleneck_multiplier: float,  # ResBottleneckBlock第一个1x1Conv维度变化倍率,默认为1,代表输出维度不变化,这样效果最好
        se_ratio: Optional[float] = None,  # SE倍率
        stage_index: int = 0,  # 阶段的id,一共4个stage 1 2 3 4
    ) -> None:
        super().__init__()

        # 重复添加
        for i in range(depth):
            block = block_constructor(
                width_in
                if i == 0
                else width_out,  # 第一个完成 width_in => width_out 其余的维度不变,全是 width_out => width_out
                width_out,
                stride
                if i == 0
                else 1,  # 第一个ResBottleneckBlock的步长为2,宽高减半,其余为1
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,  # 第一个1x1Conv维度变化倍率,默认为1,代表输出维度不变化,这样效果最好
                se_ratio,  # SE倍率
            )

            # 添加block,名称: block1-1 block1-2
            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    """
    ResBottleneckBlock的参数
    """

    def __init__(
        self,
        depths: List[int],  # 深度
        widths: List[int],  # out_channels
        group_widths: List[int],  # 分组宽度
        bottleneck_multipliers: List[
            float
        ],  # ResBottleneckBlock第一个1x1Conv维度变化倍率,默认为1,代表输出维度不变化,这样效果最好
        strides: List[int],
        se_ratio: Optional[float] = None,  # SE倍率
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    # 创建配置文件
    @classmethod
    def from_init_params(
        cls,
        depth: int,  #
        w_0: int,  #
        w_a: float,  #
        w_m: float,  #
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (
            (
                torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT))
                * QUANT
            )
            .int()
            .tolist()
        )
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = (
            torch.diff(torch.tensor([d for d, t in enumerate(splits) if t]))
            .int()
            .tolist()
        )

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
        )

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [
            _make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)
        ]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,  # block参数
        num_classes: int = 1000,  # 分类数
        stem_width: int = 32,  # stem的out_channels
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        # stem模块
        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # stem [b, 3, h, w] => [b, 32, h/2, w/2]
        self.stem = stem_type(
            3,  # in_channels
            stem_width,  # out_channels
            norm_layer,  # BN
            activation,  # ReLU
        )

        # stage开始维度
        current_width = stem_width

        # [("block1", AnyStage), ("block2", AnyStage), ("block3", AnyStage), ("block4", AnyStage)]
        blocks = []
        for i, (
            width_out,  # out_channels
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,  # in_channels
                        width_out,  # out_channels
                        stride,
                        depth,  # 重复次数
                        block_type,  # ResBottleneckBlock
                        norm_layer,  # BN
                        activation,  # ReLU
                        group_width,  # 分组宽度
                        bottleneck_multiplier,  # ResBottleneckBlock第一个1x1Conv维度变化倍率,默认为1,代表输出维度不变化,这样效果最好
                        block_params.se_ratio,  # SE倍率
                        stage_index=i + 1,  # 阶段的id,一共4个stage 1 2 3 4
                    ),
                )
            )
            # in_channels = out_channels
            current_width = width_out

        # 将Blocks放进Sequential
        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        # Init weights and good to go
        self._reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _reset_parameters(self) -> None:
        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)


def _regnet(
    arch: str,
    block_params: BlockParams,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> RegNet:
    model = RegNet(
        block_params,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1),
        **kwargs,
    )
    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def regnet_y_400mf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_400mf", params, pretrained, progress, **kwargs)


def regnet_y_800mf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_800mf", params, pretrained, progress, **kwargs)


def regnet_y_1_6gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_1_6gf", params, pretrained, progress, **kwargs)


def regnet_y_3_2gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_3_2gf", params, pretrained, progress, **kwargs)


def regnet_y_8gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_8gf", params, pretrained, progress, **kwargs)


def regnet_y_16gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=18,
        w_0=200,
        w_a=106.23,
        w_m=2.48,
        group_width=112,
        se_ratio=0.25,
        **kwargs,
    )
    return _regnet("regnet_y_16gf", params, pretrained, progress, **kwargs)


def regnet_y_32gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=20,
        w_0=232,
        w_a=115.89,
        w_m=2.53,
        group_width=232,
        se_ratio=0.25,
        **kwargs,
    )
    return _regnet("regnet_y_32gf", params, pretrained, progress, **kwargs)


def regnet_x_400mf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **kwargs
    )
    return _regnet("regnet_x_400mf", params, pretrained, progress, **kwargs)


def regnet_x_800mf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16, **kwargs
    )
    return _regnet("regnet_x_800mf", params, pretrained, progress, **kwargs)


def regnet_x_1_6gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **kwargs
    )
    return _regnet("regnet_x_1_6gf", params, pretrained, progress, **kwargs)


def regnet_x_3_2gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48, **kwargs
    )
    return _regnet("regnet_x_3_2gf", params, pretrained, progress, **kwargs)


def regnet_x_8gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120, **kwargs
    )
    return _regnet("regnet_x_8gf", params, pretrained, progress, **kwargs)


def regnet_x_16gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128, **kwargs
    )
    return _regnet("regnet_x_16gf", params, pretrained, progress, **kwargs)


def regnet_x_32gf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(
        depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168, **kwargs
    )
    return _regnet("regnet_x_32gf", params, pretrained, progress, **kwargs)


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = regnet_y_400mf()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 10]
