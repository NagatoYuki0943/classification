'''
使用的pytorch官方的代码
1.1的第一个卷积核是3x3,1.0第一个卷积核是7x7,建议使用1.1
'''

from torchvision.models import squeezenet1_0, squeezenet1_1
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Any
from torch.hub import load_state_dict_from_url


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


class Fire(nn.Module):
    '''
    挤压网络
    x => squeeze => x       降低网络深度,加快计算
    x => expand1x1 => x1
    x => expand3x3 => x3
    拼接x1和x3, 最后返回
    out_channels = expand1x1_out + expand3x3_out
    '''
    def __init__(
        self,
        inplanes: int,          # in_channels
        squeeze_planes: int,    # squeeze_channels
        expand1x1_planes: int,  # expand1x1_out_channels
        expand3x3_planes: int   # expand3x3_out_channels
    ) -> None:
        super().__init__()
        self.inplanes = inplanes

        # 降维到 squeeze_planes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        # 1x1和3x3接收squeeze的输出,最后拼接到一起
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))

        # 1x1和3x3接收squeeze的输出,最后拼接到一起
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):
    def __init__(
        self,
        version: str = '1_0',   # 版本
        num_classes: int = 1000 # 分类数
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),          # 7x7
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # in, squeeze expand1x1 expand3x3 out_channels = expand1x1 + expand3x3
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),          # 3x3
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # in, squeeze expand1x1 expand3x3    out_channels = expand1x1 + expand3x3
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                            "1_0 or 1_1 expected".format(version=version))

        # 最后的分类,使用1x1Conv代替线性层
        # Final convolution is initialized differently from the rest
        # 卷积后宽高不变然后通过平均池化的到最终结果
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))    # 高宽缩小为1
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        # [b, c, 1, 1] => [b, c]
        return torch.flatten(x, 1)


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
}


def _squeezenet(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, num_classes=1000, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    The required minimum input size of the model is 21x21.

    Args:
        num_classes: 分类数
    """
    return _squeezenet('1_0', pretrained, num_classes, **kwargs)


def squeezenet1_1(pretrained=False, num_classes=1000, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    The required minimum input size of the model is 17x17.

    Args:
        num_classes: 分类数
    """
    return _squeezenet('1_1', pretrained, num_classes, **kwargs)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = squeezenet1_1(pretrained=True)
    model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, 10, kernel_size=1)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 10]
