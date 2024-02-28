from turtle import width
from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
from torch.hub import load_state_dict_from_url


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    # 调整到离8最近的值,类似于四舍五入
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvNormActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1,
                 norm_layer=None, activation_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(nn.Conv2d(in_channels=in_planes,
                                   out_channels=out_planes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=groups,
                                   bias=False),
                         norm_layer(out_planes),
                         activation_layer(inplace=True))


class SqueezeExcitation(nn.Module):
    def __init__(self, input_c, squeeze_factor=4) -> None:
        super().__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)
    def forward(self, x:Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        # [batch, channel, 1, 1] * [batch, channel, height, width]
        # 高维度矩阵相乘是最后两个维度相乘,所以是 [1, 1] 点乘 [h, w]
        return scale * x

class InvertedResidualConfig:
    def __init__(self,
                input_c: int,
                kernel: int,
                expanded_c: int,
                out_c: int,
                use_se: bool,
                activattion: str,
                stride: int,
                width_multi: float
                ):
        self.input_c    = self.adjust_channels(input_c, width_multi)
        self.kernel     = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c      = self.adjust_channels(out_c, width_multi)
        self.use_se     = use_se
        self.use_hs     = activattion == "HS"
        self.stride     = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)

class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, norm_layer) -> None:
        super().__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.use_res_connect = (cnf.stride == 1) and (cnf.input_c == cnf.out_c)
        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # 扩展倍率因子是否为1(扩展维度是否等于in_channel), 第一层为1就不需要使用第一个 1x1 的卷积层
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvNormActivation(cnf.input_c,
                                             cnf.expanded_c,
                                             kernel_size=1,
                                             norm_layer=norm_layer,
                                             activation_layer=activation_layer))
        # depthwise DW卷积
        layers.append(ConvNormActivation(cnf.expanded_c,
                                         cnf.expanded_c,
                                         kernel_size=cnf.kernel,
                                         stride     =cnf.stride,
                                         groups     =cnf.expanded_c,
                                         norm_layer =norm_layer,
                                         activation_layer=activation_layer))
        # 注意力机制
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expanded_c))
        # project 最后的 1x1, 降维的卷积层 不使用激活函数
        layers.append(ConvNormActivation(cnf.expanded_c,
                                         cnf.out_c,
                                         kernel_size=1,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result

class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residule_setting: List[InvertedResidualConfig],
                 last_channel: int,
                 num_classes: int = 1000,
                 block            = None,
                 norm_layer       = None) -> None:
        super().__init__()

        if not inverted_residule_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")

        elif not (isinstance(inverted_residule_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residule_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # 层列表
        layers: List[InvertedResidual] = []

        #-----------------------------------#
        #   第一层卷积  V2中是32,这里改为了16
        #   stem 3x3Conv+BN+Act 调整通道,宽高减半
        #   224,224,3 -> 112,112,16
        #-----------------------------------#
        firstconv_output_c = inverted_residule_setting[0].input_c
        layers.append(ConvNormActivation(3,
                                         firstconv_output_c,
                                         kernel_size=3,
                                         stride=2,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.Hardswish))

        #-----------------------------------#
        #   重复添加15次DW卷积
        #   112,112,16 -> 7,7,160
        #-----------------------------------#
        for cnf in inverted_residule_setting:
            layers.append(block(cnf, norm_layer))

        #-----------------------------------#
        #   最后一层卷积
        #   7,7,160 -> 7,7,960
        #-----------------------------------#
        lastconv_input_c = inverted_residule_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(ConvNormActivation(lastconv_input_c,
                                         lastconv_output_c,
                                         kernel_size=1,
                                         norm_layer=norm_layer,
                                         activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # 遍历子模块 权重初始化 modules Module中的参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}

def mobilenet_v3_large(pretrained: bool = False, num_classes: int = 1000, reduced_tail: bool = False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        #                                    注意力机制
        # input_c, kernel, expand_c, out_c, use_se, activation, stride
        bneck_conf(16,  3, 16,  16, False, "RE", 1),    # 112,112,16 -> 112,112, 16

        bneck_conf(16,  3, 64,  24, False, "RE", 2),    # 112,112,16 -> 56, 56, 24 C1
        bneck_conf(24,  3, 72,  24, False, "RE", 1),

        bneck_conf(24,  5, 72,  40, True,  "RE", 2),    # C2
        bneck_conf(40,  5, 120, 40, True,  "RE", 1),    # 56, 56, 24 -> 28, 28, 40
        bneck_conf(40,  5, 120, 40, True,  "RE", 1),

        bneck_conf(40,  3, 240, 80, False, "HS", 2),    # C3
        bneck_conf(80,  3, 200, 80, False, "HS", 1),    # 28, 28, 40 -> 14, 14, 80
        bneck_conf(80,  3, 184, 80, False, "HS", 1),
        bneck_conf(80,  3, 184, 80, False, "HS", 1),

        bneck_conf(80,  3, 480, 112, True, "HS", 1),    # 14, 14, 80 -> 14, 14, 112
        bneck_conf(112, 3, 672, 112, True, "HS", 1),

        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4 14, 14, 80 -> 7, 7, 160
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)

    model = MobileNetV3(inverted_residual_setting, last_channel, num_classes)
    if pretrained:
        arch = 'mobilenet_v3_large'
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    model = mobilenet_v3_large(True)
    model.classifier[-1] = nn.Linear( model.classifier[-1].in_features, 10)
    x = torch.ones(1, 3, 224, 224)
    y = model(x)
    print(y.size())