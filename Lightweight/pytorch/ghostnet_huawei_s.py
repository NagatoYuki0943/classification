# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
华为官方实现的方法
https://github.com/huawei-noah/CV-Backbones/blob/master/ghostnet_pytorch/ghostnet.py

Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import math


__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
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
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#--------------------------------------------#
#   sigmoid(x)   = \frac 1 {1 + e^{-x}}
#   h-sigmoid(x) = \frac {ReLU6(x + 3)} {6}
#--------------------------------------------#
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super().__init__()

        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super().__init__()

        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


#----------------------------------------#
#               in
#                │
#           primary_conv(1x1Conv)
#                │
#      ┌─────────┤
#      │         │
#      │   cheap_operation(3x3DWConv)
#      │         │
#      └────────cat
#                │
#               out
#----------------------------------------#
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()

        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        #------------------------------------------------------#
        #   1x1Conv降低通道数,特征浓缩 通道数变为 oup_channel/2
        #   跨通道的特征提取
        #------------------------------------------------------#
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        #------------------------------------------------------#
        #   3x3DWConv对降低的通道数进行计算
        #   跨特征点的特诊提取
        #------------------------------------------------------#
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


#---------------#
#   倒残差结构
#---------------#
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super().__init__()

        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        #------------------------------------------------------#
        #  Point-wise 提高通道数
        #------------------------------------------------------#
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        #------------------------------------------------------#
        #   Depth-wise convolution
        #   只有步长不为1才使用
        #------------------------------------------------------#
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size-1)//2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        #------------------------------------------------------#
        #   se注意力机制
        #------------------------------------------------------#
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        #------------------------------------------------------#
        #  Point-wise 降低通道数
        #------------------------------------------------------#
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        #------------------------------------------------------#
        #   shortcut
        #   3x3DWConv -> BN -> 1x1Conv -> BN
        #------------------------------------------------------#
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super().__init__()

        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        #------------------------------------------------------#
        #   building first layer
        #   224,224,3 -> 112,112,16
        #------------------------------------------------------#
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        #------------------------------------------------------#
        #   building inverted residual blocks
        #   112,112,16 -> 7, 7,160
        #------------------------------------------------------#
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        #------------------------------------------------------#
        #   7, 7,160 -> 7, 7,960
        #------------------------------------------------------#
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        #-----------------------------------------------------------#
        #   分类层
        #   7, 7,960 -> 1,1,960 -> 1,1,1280 -> 1280 -> num_classes
        #-----------------------------------------------------------#
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(pretrained = False,**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k: 卷积核大小,表示跨特征点能力
        # t: GhostBottleneck的 mid_chs
        # c: GhostBottleneck的 out_chs
        # SE:是否使用注意力机制,0不使用
        # s: 步长
        # k, t, c, SE, s
        # stage1    114, 114, 16 -> 114, 114, 16
        [[3,  16,  16, 0, 1]],
        # stage2    114, 114, 16 -> 52, 52, 24
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3    52, 52, 24 -> 28, 28, 40
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4    28, 28, 40 -> 14, 14, 80 -> 14, 14,112
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5    14, 14, 112 ->  7, 7, 160
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    model = GhostNet(cfgs, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url('https://github.com/huawei-noah/CV-Backbones/releases/download/ghostnet_pth/ghostnet_1x.pth', progress=True)
        model.load_state_dict(state_dict)
    return model


if __name__=='__main__':
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = ghostnet(pretrained=False).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 1000]
