"""
Inception-ResNet-V2 不限制图像分辨率为299x299

https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionresnetv2.py

"""

from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ["InceptionResNetV2", "inceptionresnetv2"]


# 预训练模型的分类数是1001
pretrained_settings = {
    "inceptionresnetv2": {
        "imagenet": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth",
            "input_space": "RGB",
            "input_size": [3, 299, 299],
            "input_range": [0, 1],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "num_classes": 1000,
        },
        "imagenet+background": {
            "url": "http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth",
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
        super().__init__()
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
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):
    """
    高宽不变
    in_channels  = 192
    out_channels = 96 + 64 + 96 + 64 = 320
    """

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1),
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):
    """
    残差模块
    通道,高宽不变
    高宽: 35 x 35
    in_channels  = out_channels = 320
    branch0, branch1, branch2拼接之后交给conv2d,再和输入x相加
    """

    def __init__(self, scale=1.0):
        super().__init__()
        # scale会乘以最后的结果然后和x相加
        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1),
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # branch0, branch1, branch2拼接之后交给conv2d,再和输入x拼接
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    """
    通道,高宽变化
    kernel_size=3, stride=2
    高宽: 35 x 35 => 17 x 17
    in_channels  = 320
    out_channels = 384 + 384 + 320 = 1088
    """

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):
    """
    残差模块
    通道,高宽不变
    高宽: 17 x 17
    in_channels  = out_channels = 1088
    branch0, branch1拼接之后交给conv2d,再和输入x相加
    """

    def __init__(self, scale=1.0):
        super().__init__()
        # scale会乘以最后的结果然后和x相加
        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        # branch0, branch1拼接之后交给conv2d,再和输入x拼接
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):
    """
    通道,宽高变化
    kernel_size=3, stride=2
    宽高: 17 x 17 => 8 x 8
    in_channels  = 1088
    out_channels = 384 + 288 + 320 + 1088 = 2080
    """

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2),
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):
    """
    残差模块
    通道,高宽不变
    高宽: 8 x 8
    in_channels  = out_channels = 2080
    branch0, branch1拼接之后交给conv2d,再和输入x相加
    """

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()
        # scale会乘以最后的结果然后和x相加
        self.scale = scale
        # 是否使用ReLU,默认使用
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        # 是否使用ReLU
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)

        # branch0, branch1拼接之后交给conv2d,再和输入x相加
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1001):
        super().__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None

        # [n, 3, h, w] => [n, 192, 35, 35]
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        # [n, 192, 35, 35] => [n, 320, 35, 35]
        self.mixed_5b = Mixed_5b()
        # [n, 320, 35, 35] => [n, 320, 35, 35]  通道,高宽不变
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        # [n, 320, 35, 35] => [n, 1088, 17, 17]
        self.mixed_6a = Mixed_6a()
        # [n, 1088, 17, 17] => [n, 1088, 17, 17] 通道,高宽不变
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        # [n, 1088, 17, 17] => [n, 2080, 8, 8]
        self.mixed_7a = Mixed_7a()
        # [n, 2080, 8, 8] => [n, 2080, 8, 8] 通道,高宽不变
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        # [n, 2080, 8, 8] => [n, 2080, 8, 8] 通道,高宽不变
        self.block8 = Block8(noReLU=True)
        # [n, 2080, 8, 8] => [n, 1536, 8, 8]
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)

        # 最后的分类层
        # [n, 1536, 8, 8] => [n, 1536, 1, 1]
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        # [n, 1536] => [n, num_classes]
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        # [n, 3, h, w] => [n, 192, 35, 35]
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)

        # [n, 192, 35, 35] => [n, 320, 35, 35]
        x = self.mixed_5b(x)
        # [n, 320, 35, 35] => [n, 320, 35, 35]  通道,高宽不变
        x = self.repeat(x)
        # [n, 320, 35, 35] => [n, 1088, 17, 17]
        x = self.mixed_6a(x)
        # [n, 1088, 17, 17] => [n, 1088, 17, 17] 通道,高宽不变
        x = self.repeat_1(x)
        # [n, 1088, 17, 17] => [n, 2080, 8, 8]
        x = self.mixed_7a(x)
        # [n, 2080, 8, 8] => [n, 2080, 8, 8] 通道,高宽不变
        x = self.repeat_2(x)
        # [n, 2080, 8, 8] => [n, 2080, 8, 8] 通道,高宽不变
        x = self.block8(x)
        # [n, 2080, 8, 8] => [n, 1536, 8, 8]
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        """
        分类层
        """
        # [n, 1536, 8, 8] => [n, 1536, 1, 1]
        x = self.avgpool_1a(features)
        # [n, 1536, 1, 1] => [n, 1536]
        x = x.view(x.size(0), -1)
        # [n, 1536] => [n, num_classes]
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def inceptionresnetv2(num_classes=1000, pretrained="imagenet"):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    # 使用预训练模型
    if pretrained:
        settings = pretrained_settings["inceptionresnetv2"][pretrained]
        # 分类数必须和配置中的分类相同
        assert (
            num_classes == settings["num_classes"]
        ), "num_classes should be {}, but is {}".format(
            settings["num_classes"], num_classes
        )

        # 创建模型,加载预训练模型,预训练模型的分类数是1001
        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionResNetV2(num_classes=1001)
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
        model = InceptionResNetV2(num_classes=num_classes)
    return model


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 299, 299).to(device)
    # 预训练模型的分类数是1001
    model = inceptionresnetv2(num_classes=1001, pretrained=False)
    # pre_weights = torch.load(r"D:/AI/预训练权重/inceptionresnetv2-520b38e4.pth")
    # model.load_state_dict(pre_weights)

    model.last_linear = nn.Linear(model.last_linear.in_features, 5)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]
