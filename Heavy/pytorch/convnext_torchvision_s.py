from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param


__all__ = [
    "ConvNeXt",
    "ConvNeXt_Tiny_Weights",
    "ConvNeXt_Small_Weights",
    "ConvNeXt_Base_Weights",
    "ConvNeXt_Large_Weights",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class Permute(nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


#----------------------------------#
#   ConvNeXtBlock
#             in    [B, C, H, W]
#              │
#   ┌──────────┤
#   │      7x7DWConv
#   │          │    [B, C, H, W]
#   │      LayerNorm
#   │          │
#   │       1x1Conv
#   │          │    [B, 4C, H, W]
#   │         GELU
#   │          │
#   │       1x1Conv
#   │          │    [B, C, H, W]
#   │      LayerScale
#   │          │
#   │       DropPath
#   │          │
#   └─────────add
#              │
#             out
#----------------------------------#
class CNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            # DWConv
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),   # [B, C, H, W] -> [B, C, H, W]
            Permute([0, 2, 3, 1]),                                                  # [B, C, H, W] -> [B, H, W, C]
            norm_layer(dim),        # LayerNorm 和 Linear 默认在最后通道计算
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),            # [B, H, W, C] -> [B, H, W,4C]
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),            # [B, H, W,4C] -> [B, H, W, C]
            Permute([0, 3, 1, 2]),                                                  # [B, H, W, C] -> [B, C, H, W]
        )

        # 通道的权重 [C, 1, 1]
        # pytorch两个矩阵数据相乘,假设其中一个矩阵形状为[B, C, H, W],另一个矩阵从后往前有确定形状
        # 指示(通道可以为1)即可相乘比如[1], [W], [H, 1], [C, 1, 1], [1, C, 1, 1],更高维数据也是如此
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,            # block重复次数
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        #-----------------------------------------#
        #   Stem k=4 s=4
        #   [B, 3, 224, 224] -> [B, 96, 56, 56]
        #-----------------------------------------#
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        #-------------------------------------------------------------------------------#
        #   stages
        #   [B, 96, 56, 56] -> [B, 192, 28, 28] -> [B, 384, 14, 14] -> [B, 768, 7, 7]
        #-------------------------------------------------------------------------------#
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            #---------------------------#
            #   num_layers 次 CNBlock
            #---------------------------#
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            #------------------------------------------#
            #   后面添加下Downsampling,最后stage不添加
            #   Conv k=2 s=2
            #------------------------------------------#
            if cnf.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)    # [B, 3, 224, 224] -> [B, 768, 7, 7]
        x = self.avgpool(x)     # [B, 768, 7, 7] -> [B, 768, 1, 1]
        x = self.classifier(x)  # [B, 768, 7, 7] -> [B, 768] -> [B, num_classes]
        return x


def _convnext(
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ConvNeXt:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


_COMMON_META = {
    "min_size": (32, 32),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
    "_docs": """
        These weights improve upon the results of the original paper by using a modified version of TorchVision's
        `new training recipe
        <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
    """,
}


class ConvNeXt_Tiny_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=236),
        meta={
            **_COMMON_META,
            "num_params": 28589128,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.520,
                    "acc@5": 96.146,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Small_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_small-0c510722.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=230),
        meta={
            **_COMMON_META,
            "num_params": 50223688,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.616,
                    "acc@5": 96.650,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Base_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 88591464,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.062,
                    "acc@5": 96.870,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Large_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 197767336,
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.414,
                    "acc@5": 96.976,
                }
            },
        },
    )
    DEFAULT = IMAGENET1K_V1


def convnext_tiny(*, weights: Optional[ConvNeXt_Tiny_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Tiny_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


def convnext_small(*, weights: Optional[ConvNeXt_Small_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Small_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


def convnext_base(*, weights: Optional[ConvNeXt_Base_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Base_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


def convnext_large(*, weights: Optional[ConvNeXt_Large_Weights] = None, progress: bool = True, **kwargs: Any) -> ConvNeXt:
    weights = ConvNeXt_Large_Weights.verify(weights)

    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 1000]

    # 预处理方式
    transform = ConvNeXt_Tiny_Weights.DEFAULT.transforms()
    print(transform)
    # ImageClassification(
    #     crop_size=[224]
    #     resize_size=[236]
    #     mean=[0.485, 0.456, 0.406]
    #     std=[0.229, 0.224, 0.225]
    #     interpolation=InterpolationMode.BILINEAR
    # )

    # 查看结构
    if False:
        onnx_path = 'convnext_tiny.onnx'
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=['images'],
            output_names=['classes'],
        )
        import onnx
        from onnxsim import simplify

        # 载入onnx模型
        model_ = onnx.load(onnx_path)

        # 简化模型
        model_simple, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simple, onnx_path)
        print('finished exporting ' + onnx_path)
