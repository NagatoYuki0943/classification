"""
InceptionV4 不限制图像分辨率为299x299

开始的3层卷积和Mixed_3a Mixed_4a Mixed_5a是开始的stem模块
Inception_A Inception_B Inception_C 的维度不会发生变化
Reduction_A Reduction_B 在Inception中间起到维度变化的作用

https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py
"""

from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ["InceptionV4", "inceptionv4"]


# 预训练模型的分类数是1001
pretrained_settings = {
    "inceptionv4": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth",
            "input_space": "RGB",
            "input_size": [3, 299, 299],
            "input_range": [0, 1],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "num_classes": 1000,
        },
        "imagenet+background": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth",
            "input_space": "RGB",
            "input_size": [3, 299, 299],
            "input_range": [0, 1],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "num_classes": 1001,
        },
    }
}


class BasicConv2d(nn.Module):
    """
    Conv2d + BN + ReLU
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # verify bias false,使用BN就将bias设置为False,不然浪费性能

        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):
    """
    stem模块
    拼接 max_pool和Conv,高宽维度变化
    kernel_size=3, stride=2
    in_channels  = 64
    out_channels = 64 + 96 = 160
    """

    def __init__(self):
        super(Mixed_3a, self).__init__()

        #                          kernel_size
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):
    """
    stem模块
    拼接 Conv1和Conv2,高宽维度变化
    kernel_size=3, stride=1
    in_channels  = 160
    out_channels = 96 * 2 = 192
    """

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):
    """
    stem模块
    拼接 Conv和max_pool,高宽维度变化
    kernel_size=3, stride=2
    in_channels  = 192
    out_channels = 192 * 2 = 384
    """

    def __init__(self):
        super().__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):
    """
    通道,高宽不变
    高宽: 35 x 35
    in_channels  = 384
    out_channels = 96 * 4 = 384
    """

    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # out_channels = 96 * 4 = 384
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):
    """
    通道,高宽变化
    kernel_size=3, stride=2
    高宽: 35 x 35 => 17 x 17
    in_channels  = 384
    out_channels = 384 + 256 + 384 = 1024
    """

    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # out_channels = 384 + 256 + 384 = 1024
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):
    """
    通道,高宽不变
    高宽: 17 x 17
    in_channels  = 1024
    out_channels = 384 + 256 + 256 + 128 = 1024
    """

    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # out_channels = 384 + 256 + 256 + 128 = 1024
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):
    """
    通道,宽高变化
    kernel_size=3, stride=2
    宽高: 17 x 17 => 8 x 8
    in_channels  = 1024
    out_channels = 192 + 320 + 1024 = 1536
    """

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        # out_channels = 192 + 320 + 1024 = 1536
        return out


class Inception_C(nn.Module):
    """
    通道,高宽不变
    高宽: 8 x 8
    in_channels  = 1536
    out_channels = 256 + 256 *2 + 256 * 2 + 256 = 1536
    """

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        # 拼接 a b
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(
            384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        self.branch1_1b = BasicConv2d(
            384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )

        # 拼接 a b
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(
            384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )
        self.branch2_2 = BasicConv2d(
            448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        self.branch2_3a = BasicConv2d(
            512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        self.branch2_3b = BasicConv2d(
            512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)

        # 拼接 a b
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        # 拼接 a b
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)
        # out_channels = 256 + 256 *2 + 256 * 2 + 256 = 1536
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):
    def __init__(self, num_classes=1001):
        super().__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None

        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),  # 高宽变化
            BasicConv2d(32, 32, kernel_size=3, stride=1),  # 高宽变化
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 高宽不变
            Mixed_3a(),  # 64  => 160
            Mixed_4a(),  # 160 => 192
            Mixed_5a(),  # 192 => 384
            Inception_A(),  # 384 => 384   Inception_A维度不变
            Inception_A(),  # 384 => 384
            Inception_A(),  # 384 => 384
            Inception_A(),  # 384 => 384
            Reduction_A(),  # 384 => 1024  Mixed_6a
            Inception_B(),  # 1024 => 1024 Inception_B维度不变
            Inception_B(),  # 1024 => 1024
            Inception_B(),  # 1024 => 1024
            Inception_B(),  # 1024 => 1024
            Inception_B(),  # 1024 => 1024
            Inception_B(),  # 1024 => 1024
            Inception_B(),  # 1024 => 1024
            Reduction_B(),  # 1024 => 1536 Mixed_7a
            Inception_C(),  # 1536 => 1536 Inception_C维度不变
            Inception_C(),  # 1536 => 1536
            Inception_C(),  # 1536 => 1536
        )
        # 最后分类层
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        # 允许任何分辨率的图片被处理
        # Allows image of any size to be processed
        # 将宽高更改为1
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def inceptionv4(num_classes=1000, pretrained="imagenet"):
    if pretrained:
        settings = pretrained_settings["inceptionv4"][pretrained]
        # 分类数必须和配置文件中相同
        assert (
            num_classes == settings["num_classes"]
        ), "num_classes should be {}, but is {}".format(
            settings["num_classes"], num_classes
        )

        # 创建模型,加载预训练模型,预训练模型的分类数是1001
        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV4(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings["url"]))

        # 调整最后的分类层
        if pretrained == "imagenet":
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        # 设置输入内容
        model.input_space = settings["input_space"]
        model.input_size = settings["input_size"]
        model.input_range = settings["input_range"]

        # 设置均值和方差
        model.mean = settings["mean"]
        model.std = settings["std"]

    else:
        model = InceptionV4(num_classes=num_classes)
    return model


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    # 预训练模型的分类数是1001
    model = inceptionv4(num_classes=1001, pretrained=False)
    # pre_weights = torch.load(r"D:/AI/预训练权重/inceptionv4-8e4777a0.pth")
    # model.load_state_dict(pre_weights)
    model.last_linear = nn.Linear(1536, 5)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]
