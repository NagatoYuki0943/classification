"""
最大尺寸(渐进式学习)
S: train_size: 300, eval_size: 384
M: train_size: 384, eval_size: 480
L: train_size: 384, eval_size: 480

预训练权重在 D:\AI\预训练权重
"""

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """x为输入的张量，其通道为[B,C,H,W]，那么drop_path的含义为在一个Batch_size中，随机有drop_prob的样本，不经过主干，而直接由分支进行恒等映射
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # [B, 1, 1, 1] 随机将1条数据删除
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor  # 除以keep_prob用来保持均值不变
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


class ConvBNAct(nn.Module):
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
        super().__init__()

        # 这个padding仅适用于k为奇数的情况,偶数不适用
        padding = (kernel_size - 1) // 2
        # 默认bn
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 默认SiLU
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,  # 分组卷积
            bias=False,
        )
        # BN和激活函数
        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    """
    注意力机制
        对特征矩阵每一个channel进行池化,得到长度为channel的一维向量,使用两个全连接层,
        两个线性层的长度,最后得到权重,然后乘以每层矩阵的原值
        线性层长度变化: expand_c -> input_c / 4 -> expand_c

    fc1的输出是该Block的in_channel的四分之一
    """

    def __init__(
        self,
        input_c: int,  # MBConv 最开始通道个数
        expand_c: int,  # 经过DW卷积后的维度(就是真实输入的通道个数)
        se_ratio: float = 0.25,
    ):  # 中间层缩小倍数
        super().__init__()
        squeeze_c = int(input_c * se_ratio)
        # 两个卷积作为全连接层,kernel为1
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        x是特征矩阵
        """
        # V1使用了adaptive_avg_pool2d,这里直接在2和3维度上(高,宽)求均值
        scale = x.mean(
            (2, 3), keepdim=True
        )  # 这里的mean等同于 adaptive_avg_pool2d(x, (1, 1))
        # 使用不同的激活函数
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        # 高维度矩阵相乘是最后两个维度相乘,所以是 [h, w] 点乘 [1, 1]
        return scale * x


class FusedMBConv(nn.Module):
    """
    FusedMBConv,没有使用注意力机制, stage1,2,3
    expansion == 1 只有 3x3Conv
    expansion != 1 有   3x3Conv + 1x1Conv
    """

    def __init__(
        self,
        kernel_size: int,  # kernel
        input_c: int,  # in_channel  开始维度
        out_c: int,  # out_channel 最终维度
        expand_ratio: int,  # 第一层 1x1 Conv扩展比率
        stride: int,  # DW卷积步长 1 or 2
        se_ratio: float,  # 注意力机制,没有使用,为0
        drop_rate: float,  # MBConv最后的Dropout比率,只有在使用shortcut连接时才使用dropout层
        norm_layer,
    ):
        super().__init__()

        assert stride in [1, 2]
        assert se_ratio == 0  # 没有使用注意力机制

        # 当 stride==1 且 in_channel == out_channel 时才有shortcut连接
        self.has_shortcut = stride == 1 and input_c == out_c

        self.drop_rate = drop_rate

        # 不为1为真,说明有扩展,有没有扩展进行的运算不同
        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio  # 扩展维度

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            self.expand_conv = ConvBNAct(
                input_c,
                expanded_c,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )

            self.project_conv = ConvBNAct(
                expanded_c,
                out_c,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Identity,
            )  # 注意没有激活函数
        else:
            self.project_conv = ConvBNAct(
                input_c,
                out_c,
                kernel_size=kernel_size,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        # 有无扩展使用方法不同
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        # 是否捷径分支
        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class MBConv(nn.Module):
    """
    stage 4 5 6 [7] 使用MBConv
    1x1Conv => 3x3DWConv => 1x1Conv
    在后面的stage使用,所以不存在扩展倍率=1的情况,都是3层卷积
    最后的1x1Conv没有激活函数
    只有当stride == 1 且 in_channel == out_channel 才使用shortcut连接
    """

    def __init__(
        self,
        kernel_size: int,  # kernel 1 or 3
        input_c: int,  # in_channel  开始维度
        out_c: int,  # out_channel 最终维度
        expand_ratio: int,  # 第一层 1x1 Conv扩展比率 1 or 6
        stride: int,  # DW卷积步长 1 or 2
        se_ratio: float,  # SE模块比率,全都是0.25
        drop_rate: float,  # MBConv最后的Dropout比率, 只有在使用shortcut连接时才使用dropout层
        norm_layer: nn.Module,
    ):  # bn层
        super().__init__()

        # 判断每一层步长是否为1或2
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # shortcut连接  只有当stride == 1 且n_channel == out_channel才使用
        self.has_shortcut = stride == 1 and input_c == out_c

        # 激活函数全都是SiLU
        activation_layer = nn.SiLU  # alias Swish

        # 第一层 1x1 扩展维度
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在    stage 4 5 6 [7] 使用MBConv
        assert expand_ratio != 1

        # 第一层 1x1 扩展维度
        self.expand_conv = ConvBNAct(
            input_c,
            expanded_c,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # DW卷积 in_channel = out_channel = groups
        self.dwconv = ConvBNAct(
            expanded_c,
            expanded_c,
            kernel_size=kernel_size,
            stride=stride,
            groups=expanded_c,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        # 注意力机制                                               SE模块比率,全都是0.25
        self.se = (
            SqueezeExcite(input_c, expanded_c, se_ratio)
            if se_ratio > 0
            else nn.Identity()
        )

        # 降维 1x1 没有激活函数
        self.project_conv = ConvBNAct(
            expanded_c,
            out_c,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Identity,
        )  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)  # 注意力机制
        result = self.project_conv(result)

        # 只有在使用shortcut连接时才使用dropout层
        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class EfficientNetV2(nn.Module):
    def __init__(
        self,
        model_cnf: list,  # stage配置列表, 二维list
        num_classes: int = 1000,  # 输出个数
        num_features: int = 1280,  # stage7中 1x1 产生的channel数
        dropout_rate: float = 0.2,  # 最后全连接层前的dropout
        drop_connect_rate: float = 0.2,
    ):  # MBConv中Dropout,不是一直都是0.2,是随着网络层数增长慢慢增加的
        super().__init__()

        # 配置文件每行一共有8个元素
        for cnf in model_cnf:
            assert len(cnf) == 8

        # BN参数,使用预训练权重,必须和源作者一致
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        # ----------#
        #   stem
        # ----------#
        stem_filter_num = model_cnf[0][4]
        self.stem = ConvBNAct(
            3, stem_filter_num, kernel_size=3, stride=2, norm_layer=norm_layer
        )  # 激活函数默认是SiLU

        # -------------------------------#
        #   blocks
        #   总的重复次数,索引0是重复次数
        # -------------------------------#
        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            # 重复次数
            repeats = cnf[0]
            # 倒数第二个为0代表FusedMBConv,1代表MBConv
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            # 重复添加
            for i in range(repeats):
                blocks.append(
                    op(
                        kernel_size=cnf[1],
                        input_c=cnf[4]
                        if i == 0
                        else cnf[
                            5
                        ],  # 每个输入特征,stage中第一个卷积出入channel是上一个block的输出,其他输入channel是第一个卷积的输出(in = out)
                        out_c=cnf[5],
                        expand_ratio=cnf[3],
                        stride=cnf[2]
                        if i == 0
                        else 1,  # 每个stage第一个卷积的步长为2,其余为1
                        se_ratio=cnf[-1],
                        drop_rate=drop_connect_rate
                        * block_id
                        / total_blocks,  # drop_rate逐渐增加, 只有block_id逐渐增大
                        norm_layer=norm_layer,
                    )
                )
                block_id += 1
        # 实例化blocks
        self.blocks = nn.Sequential(*blocks)

        # -------------------------------------------------#
        #   1x1Conv + pooling + fc
        #   最后的 1x1 conv 输入是上一层输出的out_channel
        # -------------------------------------------------#
        head_input_c = model_cnf[-1][-3]
        head = OrderedDict()
        head.update(
            {
                "project_conv": ConvBNAct(
                    head_input_c, num_features, kernel_size=1, norm_layer=norm_layer
                )
            }
        )  # 激活函数默认是SiLU
        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})
        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})
        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x


def efficientnet_v2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384
    #
    # operator 0: 使用FusedMBConv 1: 代表使用MBConv
    # se_ratio: FusedMBConv=0,不使用; MBConv=0.25,使用
    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [
        [2, 3, 1, 1, 24, 24, 0, 0],
        [4, 3, 2, 4, 24, 48, 0, 0],
        [4, 3, 2, 4, 48, 64, 0, 0],
        [6, 3, 2, 4, 64, 128, 1, 0.25],
        [9, 3, 1, 6, 128, 160, 1, 0.25],
        [15, 3, 2, 6, 160, 256, 1, 0.25],
    ]

    model = EfficientNetV2(
        model_cnf=model_config, num_classes=num_classes, dropout_rate=0.2
    )
    return model


def efficientnet_v2_m(num_classes: int = 1000):
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [
        [3, 3, 1, 1, 24, 24, 0, 0],
        [5, 3, 2, 4, 24, 48, 0, 0],
        [5, 3, 2, 4, 48, 80, 0, 0],
        [7, 3, 2, 4, 80, 160, 1, 0.25],
        [14, 3, 1, 6, 160, 176, 1, 0.25],
        [18, 3, 2, 6, 176, 304, 1, 0.25],
        [5, 3, 1, 6, 304, 512, 1, 0.25],
    ]

    model = EfficientNetV2(
        model_cnf=model_config, num_classes=num_classes, dropout_rate=0.3
    )
    return model


def efficientnet_v2_l(num_classes: int = 1000):
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [
        [4, 3, 1, 1, 32, 32, 0, 0],
        [7, 3, 2, 4, 32, 64, 0, 0],
        [7, 3, 2, 4, 64, 96, 0, 0],
        [10, 3, 2, 4, 96, 192, 1, 0.25],
        [19, 3, 1, 6, 192, 224, 1, 0.25],
        [25, 3, 2, 6, 224, 384, 1, 0.25],
        [7, 3, 1, 6, 384, 640, 1, 0.25],
    ]

    model = EfficientNetV2(
        model_cnf=model_config, num_classes=num_classes, dropout_rate=0.4
    )
    return model


# efficientnet_v2 不同版本的block的stage索引
block_stages = {
    "s": [0, 2, 6, 10, 16, 25, 40],
    "m": [0, 3, 8, 13, 20, 34, 52, 57],
    "l": [0, 4, 11, 18, 28, 47, 72, 79],
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.ones(1, 3, 384, 384).to(device)

    # 只修改最后的线性层
    model = efficientnet_v2_s()
    # model.load_state_dict(pre_weights)
    model.head.classifier = nn.Linear(model.head.classifier.in_features, 5)

    model.to(device)

    # 有选择的冻住一些层
    for name, param in model.named_parameters():
        # 除最后一个卷积层和全连接层外，其他权重全部冻结
        if "head" not in name:
            # param.requires_grad = False   # same as below
            param.requires_grad_(False)
        else:
            print("training {}".format(name))
            # training head.project_conv.conv.weight
            # training head.project_conv.bn.weight
            # training head.project_conv.bn.bias
            # training head.classifier.weight
            # training head.classifier.bias

    # 需要优化的参数
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(pg, lr=0.001)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]

    # 拆解blocks
    x = model.stem(x)
    blocks = model.blocks  #  s   m   l
    print(len(blocks))  # 40  57  79

    # 获取block的stage索引
    stage = block_stages["s"]

    res = []
    for i in range(len(stage) - 1):
        x = blocks[stage[i] : stage[i + 1]](x)
        res.append(x)

    for re in res:
        print(re.size())
        # [1,  24,192,192]
        # [1,  48, 96, 96]
        # [1,  64, 48, 48]
        # [1, 128, 24, 24]
        # [1, 160, 24, 24]
        # [1, 256, 12, 12]
