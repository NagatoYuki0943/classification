'''
噼里啪啦Wz写的
torchvision的结构改了,要用新的预训练模型
SiLU == Swish

b0 224x224
b1 240x240
b2 260x260
b3 300x300
b4 380x380
b5 456x456
b6 528x528
b7 600x600
'''

import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def _make_divisible(ch, divisor=8, min_ch=None):
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
    # 最小为8
    if min_ch is None:
        min_ch = divisor
    # 调整到离8最近的值,类似于四舍五入
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """x为输入的张量，其通道为[B,C,H,W]，那么drop_path的含义为在一个Batch_size中，随机有drop_prob的样本，不经过主干，而直接由分支进行恒等映射
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # [B, 1, 1, 1] 随机将1条数据删除
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor # 除以keep_prob用来保持均值不变
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    '''
    卷积,bn,激活函数
    '''
    def __init__(self,
                 in_planes: int,        # in_channel
                 out_planes: int,       # out_channel
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,         # bn层
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # 激活函数

        # 这个padding仅适用于k为奇数的情况,偶数不适用
        padding = (kernel_size - 1) // 2
        # 默认bn
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 默认SiLU
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super().__init__(nn.Conv2d(in_channels=in_planes,
                                    out_channels=out_planes,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=groups,
                                    bias=False),
                                    norm_layer(out_planes),
                                    activation_layer())


class SqueezeExcitation(nn.Module):
    '''
    旧的写法,过程和新的一样
    注意力机制
        对特征矩阵每一个channel进行池化,得到长度为channel的一维向量,使用两个全连接层,
        两个线性层的长度,最后得到权重,然后乘以每层矩阵的原值
        线性层长度变化: expand_c -> input_c / 4 -> expand_c

    fc1的输出是该Block的in_channel的四分之一
    '''
    def __init__(self,
                 input_c: int,   # MBConv 最开始通道个数
                 expand_c: int,  # 经过DW卷积后的维度(就是真实输入的通道个数)
                 squeeze_factor: int = 4):  # 中间层缩小倍数
        super().__init__()
        squeeze_c = input_c // squeeze_factor
        # 两个卷积作为全连接层,kernel为1
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()    # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid() # sigmoid

    def forward(self, x: Tensor) -> Tensor:
        '''
        x是特征矩阵
        '''
        # [batch, channel, height, width] -> [batch, channel, 1, 1]
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # 使用不同的激活函数
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)

        # [batch, channel, 1, 1] * [batch, channel, height, width]
        # 高维度矩阵相乘是最后两个维度相乘,所以是 [1, 1] 点乘 [h, w]
        return scale * x


class InvertedResidualConfig:
    '''
    每一个stage中所有的InvertedResidual配置参数
    '''
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,           # kernel 3 or 5
                 input_c: int,          # in_channel  开始维度
                 out_c: int,            # out_channel 最终维度
                 expanded_ratio: int,   # 第一层 1x1 Conv扩展比率 1 or 6
                 stride: int,           # DW卷积步长 1 or 2
                 use_se: bool,          # 注意力机制,全都使用 True
                 drop_rate: float,      # MBConv最后的Dropout比率
                 index: str,            # 用来记录当前MBConv的名称 1a, 2a, 2b, ...
                 width_coefficient: float):# 网络宽度倍率因子
        # 调整倍率
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        '''
        调整倍率
        '''
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    '''
    倒残差
    1x1Conv => 3x3DwConv => 1x1Conv
    只有当stride == 1 且 in_channel == out_channel 才使用shortcut连接
    '''
    def __init__(self,
                 cnf: InvertedResidualConfig,   # 配置文件
                 norm_layer: Callable[..., nn.Module]): # bn层
        super().__init__()

        # 判断每一层步长是否为1或2
        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # shortcut连接  只有当stride == 1 且n_channel == out_channel才使用
        self.use_res_connect = (cnf.stride == 1) and (cnf.input_c == cnf.out_c)

        # 有序字典,顺序就是添加的顺序,默认的顺序会变化,这个不会
        layers = OrderedDict()
        # 激活函数
        activation_layer = nn.SiLU  # alias Swish

        # 扩展倍率因子是否为1(扩展维度是否等于in_channel), 第一层为1就不需要使用第一个 1x1 的卷积层 Stage2的倍率不变,所以不需要它
        if cnf.expanded_c != cnf.input_c:
            # 扩展维度
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,     # in_channel
                                                           cnf.expanded_c,  # out_channel
                                                           kernel_size=1,   # 只变换维度
                                                           norm_layer=norm_layer,   # bn层
                                                           activation_layer=activation_layer)}) # 激活层

        # depthwise DW卷积
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,   # in_channel == out_channel
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,    # groups == in_channel == out_channel
                                                  norm_layer=norm_layer,    # bn层
                                                  activation_layer=activation_layer)})  # 激活层

        # 注意力机制
        if cnf.use_se:                           # MBConv输入的维度 上层DW输出的维度
            layers.update({"se": SqueezeExcitation(cnf.input_c, cnf.expanded_c)})

        # # 1x1Conv 不要激活函数
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)}) # 最后的 1x1 卷积, 线性激活,不做任何处理

        self.block = nn.Sequential(layers)  # 卷积层
        self.out_channels = cnf.out_c       # 输出维度
        self.is_strided = cnf.stride > 1    # 步长是否为2

        # 只有在使用shortcut连接时才使用DropPath层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        # 使用捷径分支才相加
        if self.use_res_connect:
            result += x
        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,          # 网络宽度倍率因子
                 depth_coefficient: float,          # 网络深度倍率因子
                 num_classes: int = 1000,           # 分类数
                 dropout_rate: float = 0.2,         # 最后线性层之前的Dropout
                 drop_connect_rate: float = 0.2,    # MBConv中Dropout,不是一直都是0.2,是随着网络层数增长慢慢增加的
                 block: Optional[Callable[..., nn.Module]] = None,      # InvertedResidual模块
                 norm_layer: Optional[Callable[..., nn.Module]] = None  # BN
                 ):
        super().__init__()

        # B0配置表 stage2~8
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32,  16,  1, 1, True, drop_connect_rate, 1],
                       [3, 16,  24,  6, 2, True, drop_connect_rate, 2],
                       [5, 24,  40,  6, 2, True, drop_connect_rate, 2],
                       [3, 40,  80,  6, 2, True, drop_connect_rate, 3],
                       [5, 80,  112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        # 次数 = 次数 * depth_coefficient  针对 stage2~8
        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        # block默认值
        if block is None:
            block = InvertedResidual

        # norm默认是bn
        if norm_layer is None:
            # partial 为对象传入参数
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # 为adjust_channels设置默认倍率因子
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_coefficient=width_coefficient)

        # 构建配置文件,设置默认倍率因子
        bneck_conf = partial(InvertedResidualConfig, width_coefficient=width_coefficient)

        b = 0   # 统计搭建MB的次数
        # 统计重复次数,[-1]获取最后的次数,再乘以深度倍率,最后求和
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        # MBConv配置文件列表
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            # 数据复制,为了不影响源数据
            cnf = copy.copy(args)
            # 不要最后的重复次数
            for i in range(round_repeats(cnf.pop(-1))):
                # i = 0 时,stride是传入的stride,其他的stride=1
                if i > 0:
                    cnf[-3] = 1     # 将stride设置为1
                    cnf[1] = cnf[2] # input_channel = output_channel

                cnf[-1] = args[-2] * b / num_blocks  # dropout比例逐渐增大
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ... 记录顺序,通过这个方法能记录当前MBConv结构是属于第几个stage中的第几个MBConv结构
                inverted_residual_setting.append(bneck_conf(*cnf, index))   # 添加
                b += 1

        # 有序字典构建layers
        layers = OrderedDict()

        # 第一个卷积
        layers.update({"stem_conv": ConvBNActivation(in_planes=3,                   # in
                                                     out_planes=adjust_channels(32),# out
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # 构建其他卷积,名称为index = 1a 1b 2a 2b...
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})  # block是InvertedResidual

        # 最后的卷积 stage8
        # 计算最后的输入 stage7的输出
        last_conv_input_c = inverted_residual_setting[-1].out_c
        # 调整最后输出宽度
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})
        # 特征提取层
        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类层
        classifier = []
        # 设置失活比例
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # 初始化权重
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
                # 线性层 权重初始化         均值  方差
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,  # 宽度,深度不变
                        depth_coefficient=1.0,
                        dropout_rate=0.2,       # 最后失活比例
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)

    pre_weights = torch.load(r"D:/AI/预训练权重/efficientnetb0.pth")
    model = efficientnet_b0(num_classes=10)

    # 不要最后的分类层
    pre_dict ={k: v for k, v in pre_weights.items() if "classifier" not in k}
    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

    model.to(device)

    # 有选择的冻住一些层
    for name, param in model.named_parameters():
        # 除最后一个卷积层和全连接层外，其他权重全部冻结
        if ("features.top" not in name) and ("classifier" not in name):
            # param.requires_grad = False   # same as below
            param.requires_grad_(False)
        else:
            print("training {}".format(name))
            # training features.top.0.weight
            # training features.top.1.weight
            # training features.top.1.bias
            # training classifier.1.weight
            # training classifier.1.bias

    # 需要优化的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=0.001)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 10]
