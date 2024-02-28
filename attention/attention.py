"""
se注意力机制
cmba的通道和空间注意力
eca注意力机制
ca注意力机制
最后都有sigmoid激活函数
"""

import torch
import torch.nn as nn
from torch import Tensor
import math


#---------------------------------------------------#
#   se,通道注意力
#   特征层高宽全局平均池化,进行两次全连接层,第一次降低特征数,第二次还原特征数,进行sigmoid变换到0~1之间
#   最后将原值和输出结果相乘
#---------------------------------------------------#
class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio = 16,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channels, channels // ratio, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(channels // ratio, channels, bias=True),
                nn.Sigmoid()            # ❗❗❗
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        y = self.fc(y).view(b, c, 1, 1) # [B, C] -> [B, C] -> [B, C, 1, 1]
        return x * y                    # [B, C, H, W] * [B, C, 1, 1] = [B, C, H, W]


#---------------------------------------------------#
#   两个1x1Conv代替全连接层,不需要变换维度
#---------------------------------------------------#
class ConvSqueezeExcitation(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio = 16,
        activation       = nn.ReLU,
        scale_activation = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 两个卷积作为全连接层,kernel为1
        self.fc1              = nn.Conv2d(channels, channels // ratio, 1)
        self.activation       = activation()
        self.fc2              = nn.Conv2d(channels // ratio, channels, 1)
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)     # [B, C, H, W] -> [B, C, 1, 1]
        scale = self.fc1(scale)         # [B, C, H, W] -> [B, C, 1, 1]
        scale = self.activation(scale)
        scale = self.fc2(scale)         # [B, C, H, W] -> [B, C, 1, 1]
        return self.scale_activation(scale) # ❗❗❗

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input            # [B, C, 1, 1] * [B, C, H, W]


#---------------------------------------------------#
#   cbam通道注意力
#   将输入内容在宽高上分别进行平均池化和最大池化,然后经过共用的两个全连接层,然后将两个结果相加,取sigmoid,最后和原值相乘
#---------------------------------------------------#
class ChannelAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio = 16,
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc  = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 将输入内容分别进行平均池化和最大池化,然后经过共用的两个全连接层
        avg_out = self.fc(self.avg_pool(x)) # [B, C, H, W] -> [B, C, 1, 1] -> [B, C, 1, 1]
        max_out = self.fc(self.max_pool(x)) # [B, C, H, W] -> [B, C, 1, 1] -> [B, C, 1, 1]
        # 将两个结果相加,取sigmoid
        out     = avg_out + max_out         # [B, C, 1, 1] + [B, C, 1, 1] = [B, C, 1, 1]
        return self.sigmoid(out)            # ❗❗❗


#---------------------------------------------------#
#   copy from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py
#   只是用平均池化和一个1x1conv
#---------------------------------------------------#
class ChannelAttention1(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        # [B, C, H, W] * ([B, C, H, W] -> [B, C, 1, 1] -> [B, C, 1, 1]) = [B, C, H, W]
        return x * self.act(self.fc(self.pool(x)))


#---------------------------------------------------#
#   cbam空间注意力
#   不关注通道数量,所以不用给channel
#   在每一个特征点的通道上取最大值和平均值。
#   之后将这两个结果进行一个堆叠，利用一次输出通道数为1的卷积调整通道数，然后取一个sigmoid
#   在获得这个权值后，我们将这个权值乘上原输入特征层即可。
#   [B, C, H, W] -> [B, 1, H, W]
#---------------------------------------------------#
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out    = torch.mean(x, dim=1, keepdim=True) # [B, C, H, W] -> [B, 1, H, W]
        # 注意 torch.max(x, dim, keepdim=True) 返回值和下标
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, C, H, W] -> [B, 1, H, W]
        x = torch.cat([avg_out, max_out], dim=1)        # [B, 1, H, W] + [B, 1, H, W] -> [B, 2, H, W]
        x = self.conv(x)                                # [B, 2, H, W] -> [B, 1, H, W]
        return self.sigmoid(x)                          # ❗❗❗

#---------------------------------------------------#
#   cbam: 通道+空间注意力
#   先进行通道,再进行宽高
#---------------------------------------------------#
class CBMAAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio = 16,
        kernel_size = 7,
    ):
        super().__init__()
        self.channelattention = ChannelAttention(channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


#---------------------------------------------------#
#   eca
#   去除原来SE模块中的全连接层，直接在全局平均池化之后的特征上通过一个1D卷积进行学习
#   全连接层使用上一层全部数据得到下一层的全部数据
#   1D卷积使用上一层的n个数据得到下一层的全部数据,减少计算量
#---------------------------------------------------#
class ECAAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        b = 1,
        gamma = 2,
    ):
        """
        channel: 卷积通道数
        """
        super().__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1d卷积,特征长条上进行特征提取
        self.conv    = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                                            # [B, C, H, W] -> [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2).contiguous()                # [B, C, 1, 1] -> [B, C, 1] -> [B, 1, C]
        # 进行1D卷积 对C进行计算,相当于得到每个通道的权重
        y = self.conv(y).transpose(-1, -2).contiguous().unsqueeze(-1)   # [B, 1, C] -> [B, 1, C] -> [B, C, 1] -> [B, C, 1, 1]
        y = self.sigmoid(y)                                             # ❗❗❗
        return x * y.expand_as(x)


#---------------------------------------------------#
#   Coordinate Attention
#---------------------------------------------------#
class CAAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio = 16,
    ):
        super().__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channels, out_channels=channels // ratio, kernel_size=1, stride=1, bias=False)

        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channels // ratio)

        self.F_h = nn.Conv2d(in_channels=channels // ratio, out_channels=channels, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channels // ratio, out_channels=channels, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # [B, C, H, W]
        _, _, h, w = x.size()

        # [B, C, H, W] => [B, C, H, 1] => [B, C, 1, H]
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # [B, C, H, W] => [B, C, 1, H]
        x_w = torch.mean(x, dim=2, keepdim=True)

        # [B, C, 1, H] cat [B, C, 1, H] => [B, C, 1, H + W]
        x_cat = torch.cat((x_h, x_w), 3)
        # [B, C, 1, H + W] -> [B, C/R, 1, H + W]
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(x_cat)))

        # [B, C/R, 1, H + W] -> [B, C/R, 1, H], [B, C/R, 1, W]
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        # [B, C/R, 1, H] -> [B, C/R, H, 1] -> [B, C, H, 1]
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # [B, C/R, 1, W] -> [B, C, 1, W]
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        # [B, C, H, W] * [B, C, H, 1] * [B, C, 1, W] = [B, C, H, W]
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out



if __name__ == "__main__":
    x = torch.rand(1, 64, 224, 224)
    with torch.inference_mode():
        model = SqueezeExcitation(64).eval()
        print(model(x).size())  # [1, 64, 224, 224]

        model = ConvSqueezeExcitation(64, 16).eval()
        print(model(x).size())  # [1, 64, 224, 224]

        model = CBMAAttention(64).eval()
        print(model(x).size())  # [1, 64, 224, 224]

        model = ECAAttention(64).eval()
        print(model(x).size())  # [1, 64, 224, 224]

        model = CAAttention(64).eval()
        print(model(x).size())  # [1, 64, 224, 224]
