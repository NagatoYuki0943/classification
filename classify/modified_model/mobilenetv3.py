# ---------------------------------------------------#
#   MobileNet-V3: 倒残差结构
#   1x1 3x3DWConv SE 1x1
#
#   前2个巻积的s=2 -> s=1
#   因此下采样 32 -> 8
#   这两个调整不会影响预训练权重的使用
# ---------------------------------------------------#

from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
from torch.hub import load_state_dict_from_url


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    将卷积核个数(输出通道个数)调整为最接近round_nearest的整数倍,就是8的整数倍,对硬件更加友好
    ch:      输出通道个数
    divisor: 奇数,必须将ch调整为它的整数倍
    min_ch:  最小通道数
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    # 最小为8
    if min_ch is None:
        min_ch = divisor
    # 调整到离8最近的值,类似于四舍五入
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# ---------------------------------------------------#
#   Conv+BN+Act
# ---------------------------------------------------#
class ConvNormActivation(nn.Sequential):
    """
    卷积,bn,激活函数
    """

    def __init__(
        self,
        in_planes: int,  # in_channel
        out_planes: int,  # out_channel
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer=None,  # bn层
        activation_layer=None,
    ):  # 激活函数
        # 这个padding仅适用于k为奇数的情况,偶数不适用
        padding = (kernel_size - 1) // 2
        # 默认bn
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 默认relu6
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super().__init__(
            nn.Conv2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(inplace=True),
        )


# ---------------------------------------------------#
#   旧的写法,过程和新的一样
#   注意力机制
#       对特征矩阵每一个channel进行池化,得到长度为channel的一维向量,使用两个全连接层,
#       两个线性层的长度,最后得到权重,然后乘以每层矩阵的原值
#           线性层长度变化: channel -> channel / 4 -> channel
# ---------------------------------------------------#
class SqueezeExcitation(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        """
        input_c: DW卷积输出通道个数(就是输入的通道个数)
        squeeze_factor: 中间层缩小倍数
        """
        super().__init__()
        # 缩小4倍再调整为8的整数倍
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        # 两个卷积作为全连接层,kernel为1
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        x是特征矩阵
        """
        # [b, channel, height, width] -> [batch, channel, 1, 1]
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # 两个线性层的激活函数不同
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)

        # [batch, channel, 1, 1] * [batch, channel, height, width]
        # 高维度矩阵相乘是最后两个维度相乘,所以是 [1, 1] 点乘 [h, w]
        return scale * x


# ---------------------------------------------------#
#   倒残差每一层的参数
# ---------------------------------------------------#
class InvertedResidualConfig:
    def __init__(
        self,
        input_c: int,  # in_channel
        kernel: int,  # kernel 3 or 5
        expanded_c: int,  # 中间扩展channel
        out_c: int,  # out_channel
        use_se: bool,  # 注意力机制
        activation: str,  # 激活函数
        stride: int,  # DW卷积步长
        width_multi: float,
    ):  # alpha参数,调整每个通道的倍率因子
        # 调整倍率
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        self.use_hs = (
            activation == "HS"
        )  # whether using h-swish activation   使用HS或者ReLU 是一个bool
        self.stride = stride

    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        """
        调整倍率
        """
        return _make_divisible(channels * width_multi, 8)


# ---------------------------------------------------#
#   倒残差结构
#   残差:   两端channel多,中间channel少
#       降维 --> 升维
#   倒残差: 两端channel少,中间channel多
#       升维 --> 降维
#   1x1 3x3DWConv SE 1x1
#   最后的1x1Conv没有激活函数
# ---------------------------------------------------#
class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,  # 配置文件
        norm_layer,
    ):  # bn层
        super().__init__()

        # 判断每一层步长是否为1或2
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # shortcut连接  只有当stride=1且n_channel == out_channel才使用
        self.use_res_connect = (cnf.stride == 1) and (cnf.input_c == cnf.out_c)

        # 列表,每个元素都是Module类型
        layers: List[nn.Module] = []
        # 激活函数
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # 扩展倍率因子是否为1(扩展维度是否等于in_channel), 第一层为1就不需要使用第一个 1x1 的卷积层
        if cnf.expanded_c != cnf.input_c:
            # 扩展维度
            layers.append(
                ConvNormActivation(
                    cnf.input_c,  # in_channel
                    cnf.expanded_c,  # out_channel
                    kernel_size=1,  # 只变换维度
                    norm_layer=norm_layer,  # bn层
                    activation_layer=activation_layer,
                )
            )  # 激活层

        # depthwise DW卷积
        layers.append(
            ConvNormActivation(
                cnf.expanded_c,  # in_channel == out_channel == group
                cnf.expanded_c,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=cnf.expanded_c,
                norm_layer=norm_layer,  # bn层
                activation_layer=activation_layer,
            )
        )  # 激活层

        # 注意力机制
        if cnf.use_se:  # 参数是上层DW输出的维度,efficientv1中是开始输入的维度
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project 最后的 1x1, 降维的卷积层 不使用激活函数
        layers.append(
            ConvNormActivation(
                cnf.expanded_c,
                cnf.out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity,
            )
        )  # 线性激活,不做任何处理

        self.block = nn.Sequential(*layers)  # 卷积层
        self.out_channels = cnf.out_c  # 输出维度

        # ---------------------------------------------#
        #   步长大于1就为True,Lr-ASPP中使用了它
        # ---------------------------------------------#
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        # 只有使用短接才相加
        if self.use_res_connect:
            result += x

        # 最后没有激活函数,是线性激活
        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],  # 倒残差结构参数列表
        last_channel: int,  # 倒数第二个全连接层输出个数 1280 / 1024
        num_classes: int = 1000,  # 类别个数
        block=None,  # InvertedResidual模块
        norm_layer=None,
    ):  # bn
        super().__init__()

        # 如果没传入inverted_residual_setting就报错
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        # 不是list列表也报错
        # 遍历列表,是不是都是InvertedResidualConfig不然报错
        elif not (
            isinstance(inverted_residual_setting, List)
            and all(
                [
                    isinstance(s, InvertedResidualConfig)
                    for s in inverted_residual_setting
                ]
            )
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]"
            )

        # block默认值
        if block is None:
            block = InvertedResidual

        # norm默认是bn
        if norm_layer is None:
            # partial 为对象传入参数
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # 层列表
        layers: List[nn.Module] = []

        # -----------------------------------#
        #   第一层卷积  V2中是32,这里改为了16
        #   stem 3x3Conv+BN+Act 调整通道,宽高减半
        #   224,224,3 -> 112,112,16
        # -----------------------------------#
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(
            ConvNormActivation(
                3,  # rgb三通道
                firstconv_output_c,
                kernel_size=3,
                stride=1,  # 步长为2 => 1
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # -----------------------------------#
        #   重复添加15次DW卷积
        #   112,112,16 -> 7,7,160
        # -----------------------------------#
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # -----------------------------------#
        #   最后一层卷积
        #   7,7,160 -> 7,7,960
        # -----------------------------------#
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c
        layers.append(
            ConvNormActivation(
                lastconv_input_c,
                lastconv_output_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # b,960 -> b,1280 -> b,num_classes
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_c, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # 遍历子模块 权重初始化 modules Module中的参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 如果是卷积,就将bias权重设为False
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BN,将方差设置为1,均值为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 线性层 权重初始化         均值  方差
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


def mobilenet_v3_large(
    pretrained: bool = False, num_classes: int = 1000, reduced_tail: bool = False
) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    # alpha 调整out_channel
    width_multi = 1.0
    # 给InvertedResidualConfig设置默认参数,不用每次都写
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    # 对应InvertedResidualConfig中的adjust_channels方法,添加参数
    adjust_channels = partial(
        InvertedResidualConfig.adjust_channels, width_multi=width_multi
    )

    # ---------------------------------------------------#
    # True为2,False为1, 可以减少最后的全连接层参数
    # ---------------------------------------------------#
    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        #                                    注意力机制
        # input_c, kernel, expand_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),  # 112,112,16 -> 112,112, 16
        bneck_conf(
            16, 3, 64, 24, False, "RE", 1
        ),  # 112,112,16 -> 56, 56, 24 C1   s=2 => s=1
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1),  # 56, 56, 24 -> 28, 28, 40
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1),  # 28, 28, 40 -> 14, 14, 80
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),  # 14, 14, 80 -> 14, 14, 112
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(
            112, 5, 672, 160 // reduce_divider, True, "HS", 2
        ),  # C4 14, 14, 80 -> 7, 7, 160
        bneck_conf(
            160 // reduce_divider,
            5,
            960 // reduce_divider,
            160 // reduce_divider,
            True,
            "HS",
            1,
        ),
        bneck_conf(
            160 // reduce_divider,
            5,
            960 // reduce_divider,
            160 // reduce_divider,
            True,
            "HS",
            1,
        ),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5 7, 7, 160 -> 7, 7, 1280

    model = MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        num_classes=num_classes,
    )
    if pretrained:
        arch = "mobilenet_v3_large"
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        model.load_state_dict(state_dict)
    return model


def mobilenet_v3_small(
    pretrained: bool = False, num_classes: int = 1000, reduced_tail: bool = False
) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>.
    weights_link:
    https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes (int): number of classes
        reduced_tail (bool): If True, reduces the channel counts of all feature layers
            between C4 and C5 by 2. It is used to reduce the channel redundancy in the
            backbone for Detection and Segmentation.
    """
    # alpha 调整out_channel
    width_multi = 1.0
    # 给InvertedResidualConfig设置默认参数,不用每次都写
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    # 对应InvertedResidualConfig中的adjust_channels方法,添加参数
    adjust_channels = partial(
        InvertedResidualConfig.adjust_channels, width_multi=width_multi
    )

    # True为2,False为1, 可以减少最后的全连接层参数
    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        #                                    注意力机制
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1 112,112,16 -> 56, 56, 16
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),  # 56, 56, 16 -> 28, 28, 24
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),  # 28, 28, 24 -> 14, 14, 40
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(
            48, 5, 288, 96 // reduce_divider, True, "HS", 2
        ),  # C4 14, 14, 40 -> 7, 7, 96
        bneck_conf(
            96 // reduce_divider,
            5,
            576 // reduce_divider,
            96 // reduce_divider,
            True,
            "HS",
            1,
        ),
        bneck_conf(
            96 // reduce_divider,
            5,
            576 // reduce_divider,
            96 // reduce_divider,
            True,
            "HS",
            1,
        ),
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5 7, 7, 96 -> 7, 7, 1024

    model = MobileNetV3(
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        num_classes=num_classes,
    )
    if pretrained:
        arch = "mobilenet_v3_small"
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobilenet_v3_large(pretrained=True)
    # 更改最后的分类层
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)

    model.to(device)

    # 冻结卷积层,这里使用了 model.features. 如果使用 model. 就冻住所有层了
    for param in model.features.parameters():
        param.requires_grad = False

    # 打印训练权重,只有最后的线性层
    print("=" * 40 + "\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            # classifier.0.weight
            # classifier.0.bias
            # classifier.3.weight
            # classifier.3.bias
    print("=" * 40)

    # 需要优化的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=0.001)

    model.eval()
    x = torch.ones(2, 3, 32, 32).to(device)
    y = model(x)
    print(y.size())  # torch.Size([2, 5])
