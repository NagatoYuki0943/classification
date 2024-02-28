'''
直接复制pytorch官方的代码


辅助输出的形状和最终输出的形状完全相同,形状都是[batch, num_classes];只在训练时会有,验证时没有

'''

from collections import namedtuple
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Callable, Any, Optional, Tuple, List

__all__ = ['Inception3', 'inception_v3', 'InceptionOutputs', '_InceptionOutputs']


InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


class BasicConv2d(nn.Module):
    '''
    Conv + BN + ReLU
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    '''
    卷积之后大小不发生变化
    1X1Conv, 5X5Conv, 3X3Conv, avg_pool
    out_channel = 64 + 64 + 96 + pool_features
    '''
    def __init__(
        self,
        in_channels: int,
        pool_features: int, # max_pool中输出维度个数
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        # 默认使用BasicConv2d
        if conv_block is None:
            conv_block = BasicConv2d

        # 1X1Conv
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        # 5X5Conv
        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        # 3X3Conv
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        # max_pool
        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        # 1X1Conv
        branch1x1 = self.branch1x1(x)

        # 5X5Conv
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        # 3X3Conv
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # avg_pool
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        # out_channel = 64 + 64 + 96 + pool_features
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    '''
    宽高变化 kernel_size=3, stride=2
    3x3Conv1, 3x3Conv2, max_pool
    out_channels = 384 + 96 + in_channels
    '''
    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        # 默认使用BasicConv2d
        if conv_block is None:
            conv_block = BasicConv2d

        # 3x3Conv1
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        # 3x3Conv2
        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        # 3x3Conv1
        branch3x3 = self.branch3x3(x)

        # 3x3Conv2
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # max_pool
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        # out_channels = 384 + 96 + in_channels
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    '''
    宽高不变
    1x1Conv, 7x7Conv1, 7x7Conv2, avg_pool
    out_channels = 192 * 4 = 768
    '''
    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,  # 每个branch的中间维度变化
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        # 默认使用BasicConv2d
        if conv_block is None:
            conv_block = BasicConv2d

        # 1x1Conv
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        # 每个branch的中间维度变化
        c7 = channels_7x7

        # 7x7Conv1
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        # 7x7Conv2
        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7,  kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7,  kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7,  kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        # avg_pool
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        # 1x1Conv
        branch1x1 = self.branch1x1(x)

        # 7x7Conv1
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        # 7x7Conv2
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # avg_pool
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        # out_channels = 192 * 4 = 768
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    '''
    宽高变化 kernel_size=3, stride=2
    3x3Conv + 7x7Conv + max_pool2d
    out_channels = 320 + 192 + in_channels
    '''
    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        # 默认使用BasicConv2d
        if conv_block is None:
            conv_block = BasicConv2d

        # 3x3Conv
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        # 7x7Conv
        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        # 3x3Conv
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        # 7x7Conv
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        # max_pool
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        # out_channels = 320 + 192 + in_channels
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    '''
    宽高不变
    1x1Conv + 3x3Conv1拼接a和b + 3x3Conv2 拼接a和b + avg_pool
    out_channels = 320 + 384 * 2 + 384 * 2 + 192 = 2048
    '''
    def __init__(
        self,
        in_channels: int,
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        # 默认使用BasicConv2d
        if conv_block is None:
            conv_block = BasicConv2d

        # 1x1Conv
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        # 3x3Conv1 拼接a和b out_channel = 384 * 2 = 768
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # 3x3Conv2 拼接a和b out_channel = 384 * 2 = 768
        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # avg_pool
        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        # 1x1Conv
        branch1x1 = self.branch1x1(x)

        # 3x3Conv1 拼接a和b out_channel = 384 * 2 = 768
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # 3x3Conv2 拼接a和b out_channel = 384 * 2 = 768
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # avg_pool
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        # out_channels = 320 + 384 * 2 + 384 * 2 + 192 = 2048
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    '''
    辅助分类层
    辅助输出的形状和最终输出的形状完全相同,形状都是[batch, num_classes];只在训练时会有,验证时没有
    '''
    def __init__(
        self,
        in_channels: int,   # 维度
        num_classes: int,   # 分类数
        conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        # 默认使用BasicConv2d
        if conv_block is None:
            conv_block = BasicConv2d

        # [b, 768, 5, 5] => [b, 128, 5, 5]
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)

        # [b, 128, 5, 5] => [b, 768, 5, 5]
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]

        # [b, 768] => [b, num_classes]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # [b, 768, 17, 17] => [b, 768, 5, 5]
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # [b, 768, 5, 5] => [b, 128, 5, 5]
        x = self.conv0(x)
        # [b, 128, 5, 5] => [b, 768, 5, 5]
        x = self.conv1(x)
        # [b, 768, 5, 5] => [b, 768, 1, 1]
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # [b, 768, 1, 1] => [b, 768]
        x = torch.flatten(x, 1)
        # [b, 768] => [b, num_classes]
        x = self.fc(x)
        return x


class Inception3(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000,                # 分类数
        aux_logits: bool = True,                # 辅助分类层
        transform_input: bool = False,          # 使用ImageNet均值化处理图像
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,  # 不同层使用的卷积类
        init_weights: Optional[bool] = None     # 初始化权重
    ) -> None:
        super(Inception3, self).__init__()

        # 规定每层使用哪一个卷积类
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True

        # Inception_blocks必须是7层
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        # 确定是否使用辅助分类函数
        self.aux_logits = aux_logits
        # 使用ImageNet均值化处理图像
        self.transform_input = transform_input
        # N x 3 x 299 x 299
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        # N x 32 x 149 x 149
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        # N x 32 x 147 x 147
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        # N x 64 x 147 x 147
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        # N x 80 x 73 x 73
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        # N x 192 x 71 x 71
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # N x 192 x 35 x 35

        # 3个
        self.Mixed_5b = inception_a(192, pool_features=32)
        # N x 256 x 35 x 35
        self.Mixed_5c = inception_a(256, pool_features=64)
        # N x 288 x 35 x 35
        self.Mixed_5d = inception_a(288, pool_features=64)
        # N x 288 x 35 x 35

        # 5个
        self.Mixed_6a = inception_b(288)
        # N x 768 x 17 x 17
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        # N x 768 x 17 x 17
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        # N x 768 x 17 x 17
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        # N x 768 x 17 x 17
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        # N x 768 x 17 x 17

        # 辅助输出函数,使用的是 Mixed_6e 的输出
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)

        # 3个
        # N x 768 x 17 x 17
        self.Mixed_7a = inception_d(768)
        # N x 1280 x 8 x 8
        self.Mixed_7b = inception_e(1280)
        # N x 2048 x 8 x 8
        self.Mixed_7c = inception_e(2048)
        # N x 2048 x 8 x 8
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        # 初始化权重
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, 'stddev') else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        '''
        使用ImageNet均值化处理图像
        '''
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17

        # 辅助输出函数,使用的是 Mixed_6e 的输出
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8

        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        # 只有在训练模式并且aux_logits为True才使用辅助输出函数
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            # eval模式只返回最后的输出
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        # 使用ImageNet均值化处理图像
        x = self._transform_input(x)

        x, aux = self._forward(x)

        # 只有在训练模式并且aux_logits为True才使用辅助输出函数
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
}

def inception_v3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Inception3:
    r"""
    The required minimum input size of the model is 75x75.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: True if ``pretrained=True``, else False.
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" in kwargs:
            original_aux_logits = kwargs["aux_logits"]
            kwargs["aux_logits"] = True
        else:
            original_aux_logits = True
        kwargs["init_weights"] = False  # we are loading weights from a pretrained model
        model = Inception3(**kwargs)
        state_dict = load_state_dict_from_url(model_urls["inception_v3_google"], progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None
        return model

    return Inception3(**kwargs)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(2, 3, 299, 299).to(device)
    model = inception_v3(pretrained=False).to(device)

    model.eval()
    with torch.inference_mode():
        y, aux = model(x)
    print(y.size())
    print(aux.size())
    # torch.Size([1000])
    # torch.Size([1000])
