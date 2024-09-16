# ---------------------------------------------------#
#   MobileNet-V2: 倒残差结构
#   1x1 3x3DWConv 1x1
# ---------------------------------------------------#


import torch
from torch import nn
from torch.nn import functional as F
from typing import Any
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
    卷积层 = 卷积+ BN + ReLU6
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        """
        groups: 默认为1,就是普通卷积,和ResNeXt相同,如果和in_channel相同,就是DW卷积
        """
        # 这个padding仅适用于k为奇数的情况,偶数不适用
        padding = (kernel_size - 1) // 2

        # 继承自Sequential只需要在super().__init__()中添加序列即可
        super().__init__(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )


# ---------------------------------------------------#
#   倒残差结构
#   残差:   两端channel多,中间channel少
#       降维 --> 升维
#   倒残差: 两端channel少,中间channel多
#       升维 --> 降维
#   1x1 3x3DWConv 1x1
#   最后的1x1Conv没有激活函数
# ---------------------------------------------------#
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        """
        expand_ratio: 扩展因子,表格中的t
        """
        super().__init__()

        # 第一层卷积核个数,第一层输出维度
        hidden_channel = in_channel * expand_ratio
        # 步长为1同时通道不变化才相加
        self.use_shortcut = (stride == 1) and (in_channel == out_channel)

        layers = []
        # ----------------------------------------------------#
        #   利用1x1卷积根据输入进来的通道数进行通道数上升,不扩张就不需要第一个1x1卷积了
        # ----------------------------------------------------#
        if expand_ratio != 1:
            layers.append(ConvNormActivation(in_channel, hidden_channel, kernel_size=1))
        layers.extend(
            [
                # --------------------------------------------#
                #   进行3x3的DW卷积
                # --------------------------------------------#
                ConvNormActivation(
                    hidden_channel, hidden_channel, stride=stride, groups=hidden_channel
                ),
                # -----------------------------------#
                #   利用1x1卷积进行通道数的调整,没有激活函数
                # -----------------------------------#
                nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        """
        alpha: 控制卷积核个数倍率
        """
        super().__init__()

        # 使用倒残差结构
        block = InvertedResidual
        # 第一层输入的个数   将卷积核个数(输出通道个数)调整为round_nearest的整数倍 就是调整为8个整数倍
        input_channel = _make_divisible(32 * alpha, round_nearest)
        # 最后通道个数
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t 扩展因子,第一层卷积让channel维度变多
            # c out_channel 或 k1
            # n bottlenet 重复次数
            # s 步长(每一个block的第一层步长)
            # t, c, n, s
            [1, 16, 1, 1],  # 112,112,32 -> 112,112,16
            [6, 24, 2, 2],  # 112,112,16 ->  56, 56,24
            [6, 32, 3, 2],  #  56, 56,24 ->  28, 28,32
            [6, 64, 4, 2],  #  28, 28,32 ->  14, 14,64
            [6, 96, 3, 1],  #  14, 14,64 ->  14, 14,96
            [6, 160, 3, 2],  #  14, 14,96 ->  7, 7, 160
            [6, 320, 1, 1],  #  7, 7, 160 ->  7, 7, 320
        ]

        # 卷积层
        features = []

        # -----------------------------------#
        #   第一层卷积
        #   224,224,3 -> 112,112,32
        # -----------------------------------#
        features.append(ConvNormActivation(3, input_channel, stride=2))

        # -----------------------------------#
        #   重复添加n次DW卷积
        #   112,112,32 -> 7,7,320
        # -----------------------------------#
        for t, c, n, s in inverted_residual_setting:
            # 调整输出个数
            output_channel = _make_divisible(c * alpha, round_nearest)
            # 添加n次
            for i in range(n):
                # stride只有第一次为2,其余的为1
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                # 调整下一层的输入channel
                input_channel = output_channel

        # -----------------------------------#
        #   最后一层卷积
        #   7,7,320 -> 7,7,1280
        # -----------------------------------#
        features.append(ConvNormActivation(input_channel, last_channel, 1))

        self.features = nn.Sequential(*features)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(last_channel, num_classes)
        )

        # 遍历子模块 权重初始化 modules Module中的参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 如果是卷积,就将bias权重设为False
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # BN,将方差设置为1,均值为0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 线性层 权重初始化       均值  标准差
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # 变为 二维
        # x = x.view(x.size(0), -1)
        # x = torch.flatten(x, 1)
        x = x.flatten(1)

        x = self.classifier(x)
        return x


model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


def mobilenet_v2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> MobileNetV2:
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["mobilenet_v2"], progress=progress
        )
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = mobilenet_v2(pretrained=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)
    model = model.to(device)

    # 打印训练权重,只有最后的线性层
    for name, param in model.named_parameters():
        if "features" in name:
            param.requires_grad = False
        else:
            print(name)
            # classifier.1.weight
            # classifier.1.bias

    # 需要优化的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=0.001)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]
