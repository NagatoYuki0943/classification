"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


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
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -----------------------------------------------------------#
#   Transformer中使用的Layer Normalization（LN）
#   pytorch的LayerNorm默认是对最后的维度进行计算,如果channel放在最后可以使用pytorch自带的,
#   如果channel在前面就用这个
# -----------------------------------------------------------#
class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            # -----------------------------------------------------------#
            #   通道在最后就使用pytorch官方的方法
            # -----------------------------------------------------------#
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            # [b, c, h, w]
            #   1. 对channel求均值
            mean = x.mean(1, keepdim=True)
            #   2. (x-mean)^2再求均值得到方法
            var = (x - mean).pow(2).mean(1, keepdim=True)
            #   3. (x-mean)/ 标准差
            x = (x - mean) / torch.sqrt(var + self.eps)
            #   4. 最后乘以权重加上偏置
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# -----------------------------------------------------------#
#   ConvNextBlock 改进的残差块,通道宽高不变:   DWConv + 1x1Conv + 1x1Conv;  ReLU -> GeLU;  BN -> LN;  更少的激活函数和LN;  Layer Scale
#      DWConv k=7,s=1,p=3  c -> c
# ​     LN
# ​     1x1Conv             c -> 4c     作者使用Linear实现的
# ​     GELU
# ​     1x1Conv             4c -> c     作者使用Linear实现的
# ​     Layer Scale         对每一个通道数据单独进行缩放,乘以一个可训练的系数
# ​     Drop Path
#   最终残差相加
# -----------------------------------------------------------#
class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (B, C, H, W)
    (2) DwConv -> Permute to (B, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        # -----------------------------------------------------------#
        #   DWConv k=7,s=1,p=3  c -> c
        # -----------------------------------------------------------#
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        # -----------------------------------------------------------#
        #   channels_last: 使用pytorch自带的LN
        # -----------------------------------------------------------#
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # -----------------------------------------------------------#
        #   1x1Conv             c -> 4c     作者使用Linear实现的,不用转换为二维数据就能运行
        # -----------------------------------------------------------#
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        # -----------------------------------------------------------#
        #   1x1Conv             4c -> c     作者使用Linear实现的,不用转换为二维数据就能运行
        # -----------------------------------------------------------#
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # -----------------------------------------------------------#
        #   Layer Scale: 对每一个通道数据进行缩放
        #   参数数量和数据通道数相等
        # -----------------------------------------------------------#
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim,)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        # -----------------------------------------------------------#
        #   Layer Scale: 对每一个通道数据单独进行缩放,乘以一个可训练的系数
        # -----------------------------------------------------------#
        self.drop_path = DropPath(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: list = None,  # 每个stage中Block重复次数
        dims: list = None,  # 每个stage中Block的in_channels
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ):
        super().__init__()

        # -----------------------------------------------------------#
        #   stem部分 + 3次downsample
        #   把stem当做stage1的下采样部分
        # -----------------------------------------------------------#
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers

        # -----------------------------------------------------------#
        #   stem部分: Conv k=4 s=4 + LN
        #   224,224,3 -> 56,56,96
        # -----------------------------------------------------------#
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # -----------------------------------------------------------#
        #   对应stage2-stage4前的3个downsample 通道翻倍,宽高减半
        #   LN + Conv k=2 s=2
        #   downsample2: 56,56,96  -> 28,28,192
        #   downsample3: 28,28,192 -> 14,14,384
        #   downsample4: 14,14,384 -> 7, 7, 768
        # -----------------------------------------------------------#
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # -----------------------------------------------------------#
        #   4个stage
        #   重复堆叠n次ConvNextBlock [3, 3, 9, 3]
        #   stage1的进出形状: 56,56,96
        #   stage2的进出形状: 28,28,192
        #   stage3的进出形状: 14,14,384
        #   stage4的进出形状: 7, 7, 768
        # -----------------------------------------------------------#
        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple blocks

        # -----------------------------------------------------------#
        #   构造等差数列,总共有所有Block个数,从0到drop_path_rate
        # -----------------------------------------------------------#
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_rate=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            # 这里是计数,防止上面的j取到的都是开始的数据
            cur += depths[i]

        # -----------------------------------------------------------#
        #   平均池化 + LN + Linear
        # -----------------------------------------------------------#
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        # -----------------------------------------------------------#
        #   初始化
        # -----------------------------------------------------------#
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            # downsample: stem + 3次下采样
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)    mean默认会减少维度,两次就变为二维数据

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def convnext_tiny(num_classes: int = 1000, pretrained: bool = False):
    model = ConvNeXt(
        depths=[3, 3, 9, 3],  # 每个stage中Block重复次数
        dims=[96, 192, 384, 768],  # 每个stage中Block的in_channels
        num_classes=num_classes,
    )
    if pretrained:
        url = model_urls["convnext_tiny_1k"]
        state_dict = load_state_dict_from_url(url)[
            "model"
        ]  # 注意这个字典有model,和其他模型不同
        model.load_state_dict(state_dict)
    return model


def convnext_small(num_classes: int = 1000, pretrained: bool = False):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],  # 每个stage中Block重复次数
        dims=[96, 192, 384, 768],  # 每个stage中Block的in_channels
        num_classes=num_classes,
    )
    if pretrained:
        url = model_urls["convnext_small_1k"]
        state_dict = load_state_dict_from_url(url)[
            "model"
        ]  # 注意这个字典有model,和其他模型不同
        model.load_state_dict(state_dict)
    return model


def convnext_base(
    num_classes: int = 1000, pretrained: bool = False, in_22k: bool = False
):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],  # 每个stage中Block重复次数
        dims=[128, 256, 512, 1024],  # 每个stage中Block的in_channels
        num_classes=num_classes,
    )
    if pretrained:
        url = (
            model_urls["convnext_base_22k"]
            if in_22k
            else model_urls["convnext_base_1k"]
        )
        state_dict = load_state_dict_from_url(url)[
            "model"
        ]  # 注意这个字典有model,和其他模型不同
        model.load_state_dict(state_dict)
    return model


def convnext_large(
    num_classes: int = 1000, pretrained: bool = False, in_22k: bool = False
):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],  # 每个stage中Block重复次数
        dims=[192, 384, 768, 1536],  # 每个stage中Block的in_channels
        num_classes=num_classes,
    )
    if pretrained:
        url = (
            model_urls["convnext_large_22k"]
            if in_22k
            else model_urls["convnext_large_1k"]
        )
        state_dict = load_state_dict_from_url(url)[
            "model"
        ]  # 注意这个字典有model,和其他模型不同
        model.load_state_dict(state_dict)
    return model


def convnext_xlarge(num_classes: int = 1000, pretrained: bool = False):
    url = "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth"
    model = ConvNeXt(
        depths=[3, 3, 27, 3],  # 每个stage中Block重复次数
        dims=[256, 512, 1024, 2048],  # 每个stage中Block的in_channels
        num_classes=num_classes,
    )
    if pretrained:
        url = model_urls["convnext_xlarge_22k"]
        state_dict = load_state_dict_from_url(url)[
            "model"
        ]  # 注意这个字典有model,和其他模型不同
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    model = convnext_tiny(pretrained=True)
    model.head = nn.Linear(model.head.in_features, 5)
    x = torch.ones(1, 3, 224, 224)
    y = model(x)
    print(y.size())  # torch.Size([1, 5])
