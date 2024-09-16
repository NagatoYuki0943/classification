from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """
    将特征矩阵平均分成2组,将每个组的数据进一步划分为2块,再将相同块的放到一起
    group1    group2
    a a a a   b b b b   这些都是不同的通道

        a         b
        a         b
        a         b
        a         b

            a  a  a	a
            b  b  b	b

    a b a b a b a b   交换的通道的位置,不是每个通道内部的数据
    """
    # b c h w
    batch_size, num_channels, height, width = x.size()
    # 每个组合的通道个数
    channels_per_group = num_channels // groups

    # 改变形状,增加1个维度 num_channels => groups, channels_per_group
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # 让不同group中的相同序号的channel到同一个维度
    # groups, channels_per_group 交换位置
    x = x.transpose(
        1, 2
    ).contiguous()  # contiguous() 将tensor数据转化为内存中连续的数据

    # flatten 将不同的组的相同序号的channel连接到一起
    x = x.view(batch_size, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    """
    倒残差结构
    stride = 1: 输入channel一分为二,左侧不计算,右侧计算, 最后拼接到一起,维度不变 in_channel == out_channel == branch_features*2
    stride = 2: 输入维度是 input,两侧都计算,开始维度没有一分为二,不过中间降维一般,不影响后面的拼接, in_channel != out_channel == branch_features*2
    """

    def __init__(self, input_c: int, output_c: int, stride: int):
        super().__init__()

        # 步长必须为1或2
        if stride not in [1, 2]:
            raise ValueError("illegal stride value, stride should 1 or 2.")
        self.stride = stride

        # out_channel必须是2的整数倍
        assert output_c % 2 == 0
        # 左右两个分支维度是输出维度一半,最后拼接到一起
        branch_features = output_c // 2

        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        # 前面为True时直接满足了,前面不为True(就是stride=1)时,in_channel == out_channel = branch_features * 2
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            # 步长为2,左右分支都有
            self.branch1 = nn.Sequential(
                # 输入维度就是in_channel
                self.depthwise_conv(
                    input_c, input_c, kernel_s=3, stride=self.stride, padding=1
                ),  # DW卷积,输入和输出不变 3x3
                nn.BatchNorm2d(input_c),
                nn.Conv2d(
                    input_c,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),  # 1x1
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            # 步长为1,左分支没有
            self.branch1 = nn.Sequential()

        # 右分支
        self.branch2 = nn.Sequential(
            # stride = 1 输入维度是 output / 2, 因为最后开始将channel一分为二,后面要拼接, in_channel == out_channel == branch_features*2
            # stride = 2 输入维度是 input, 开始维度没有一分为二,不过中间降维了,不影响后面的拼接, in_channel != out_channel == branch_features*2
            nn.Conv2d(
                input_c if self.stride > 1 else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),  # 1x1
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_s=3,
                stride=self.stride,
                padding=1,
            ),  # 3x3
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),  # 1x1
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    # DW卷积 groups == in_channel == out_channel
    @staticmethod
    def depthwise_conv(
        input_c: int,
        output_c: int,
        kernel_s: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> nn.Conv2d:
        return nn.Conv2d(
            in_channels=input_c,
            out_channels=output_c,
            kernel_size=kernel_s,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=input_c,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            # 步长为1, 就平均划分为两份,最后拼接,最后维度不变
            x1, x2 = x.chunk(chunks=2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            # 步长为2,不划分,都进行计算,维度减半,然后拼接到一起,最后维度不变
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        # 每次拼接完成才重新混淆数据
        out = channel_shuffle(out, groups=2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stages_repeats: List[int],  # stage2 3 4的重复次数
        stages_out_channels: List[
            int
        ],  # conv1 stage2 3 4 5 的输出维度(每个阶段的第一个卷积就变化)
        num_classes: int = 1000,  # 分类
        inverted_residual: Callable[..., nn.Module] = InvertedResidual,
    ):  # 卷积类
        super().__init__()

        # stage 2 3 4 都有重复,必须是3个数据
        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        # conv1 stage2 3 4 5 必须有5个输出维度
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        # conv1的输出维度
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # stage2的in_channel
        input_channels = output_channels
        # Conv1下面紧邻的MaxPool
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3个stage
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        # 遍历搭建stage2 3 4                        name         [4, 8, 4]           [116, 232, 464]
        for name, repeats, output_channels in zip(
            stage_names, stages_repeats, self._stage_out_channels[1:]
        ):
            # stage中第一个conv的stride = 2
            seq = [inverted_residual(input_channels, output_channels, stride=2)]
            # 下面的其他的conv
            for i in range(repeats - 1):
                # 步长为1的情况下 in_channel == out_channel
                seq.append(
                    inverted_residual(output_channels, output_channels, stride=1)
                )
            # 给类设置变量
            setattr(self, name, nn.Sequential(*seq))
            # 重复变化in_channel
            input_channels = output_channels

        # 最后的conv5  输出维度是最后的维度
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x0_5(num_classes=1000, pretrained=False):
    """
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(
        stages_repeats=[4, 8, 4],  # stage2 3 4的重复次数
        stages_out_channels=[
            24,
            48,
            96,
            192,
            1024,
        ],  # conv1 stage2 3 4 5 的输出维度(每个阶段的第一个卷积就变化)
        num_classes=num_classes,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            progress=True,
        )
        model.load_state_dict(state_dict)
    return model


def shufflenet_v2_x1_0(num_classes=1000, pretrained=False):
    """
    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(
        stages_repeats=[4, 8, 4],  # stage2 3 4的重复次数
        stages_out_channels=[
            24,
            116,
            232,
            464,
            1024,
        ],  # conv1 stage2 3 4 5 的输出维度(每个阶段的第一个卷积就变化)
        num_classes=num_classes,
    )
    if pretrained:
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/shufflenetv2_x0.5-5666bf0f80.pth",
            progress=True,
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
    model = shufflenet_v2_x0_5(pretrained=False).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 1000]
