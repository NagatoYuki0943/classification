# https://github.com/Pritam-N/ParNet/blob/main/parnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict


def activation_func(activation):
    return nn.ModuleDict(
        [
            ["relu", nn.ReLU(inplace=True)],
            ["silu", nn.SiLU(inplace=True)],
            ["none", nn.Identity()],
        ]
    )[activation]


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


def conv_sampler(kernel_size=1, stride=1, groups=1):
    return partial(
        Conv2dAuto, kernel_size=kernel_size, stride=stride, bias=False, groups=groups
    )


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(
        OrderedDict(
            {
                "çonv": conv(in_channels, out_channels, *args, **kwargs),
                "bn": nn.BatchNorm2d(out_channels),
            }
        )
    )


class Conv2dBN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, stride=1, groups=1
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


def conv_van(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(
        OrderedDict(
            {
                "çonv": conv(in_channels, out_channels, *args, **kwargs),
            }
        )
    )


# class GlobalAveragePool2D():
#     def __init__(self, keepdim=True) -> None:
#         # super(GlobalAveragePool2D, self).__init__()
#         self.keepdim = keepdim

#     # def forward(self, inputs):
#     #     return torch.mean(inputs, axis=[2, 3], keepdim=self.keepdim)

#     def __call__(self, inputs, *args, **kwargs):
#         return torch.mean(inputs, axis=[2, 3], keepdim=self.keepdim)


# -------------------------------------------#
#                    │
#       ┌────────────┼────────────┐
#       │            │            │
#   2x2Avgpool       │    AdaptiveAvgPool2d
#       │            │            │
#    1x1Conv      3x3Conv      1x1Conv
#       │           s=2           │
#       │            │            │
#      BN           BN         Sigmoid
#       │            │            │
#       └───────────add           │
#                    │            │
#                   mul───────────┘
#                    │
#                   SiLU
#                    │
# -------------------------------------------#
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Downsample, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.avgpool = nn.AvgPool2d(
            2, ceil_mode=True
        )  # ceil_mode=True: fix the size mismatch in x and y
        # self.conv1 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=1))
        # self.conv2 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=3, stride=2))
        # self.conv3 = conv_van(self.in_channels, self.out_channels, conv_sampler(kernel_size=1))

        self.conv1 = Conv2dBN(in_channels, out_channels, kernel_size=1)
        self.conv2 = Conv2dBN(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)  # GlobalAveragePool2D
        self.activation = activation_func("silu")
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 分支1
        x = self.avgpool(inputs)  # [1, 3, 224, 224] -> [1, 3, 112, 112]
        x = self.conv1(x)  # [1, 3, 112, 112] -> [1, 64, 112, 112]

        # 分支2
        y = self.conv2(inputs)  # [1, 3, 224, 224] -> [1, 64, 112, 112]

        # 分支3
        z = self.globalAvgPool(inputs)  # [1, 3, 224, 224] -> [1, 3, 1, 1]
        z = self.conv3(z)  # [1, 3, 1, 1] -> [1, 64, 1, 1]
        z = self.sigmoid(z)

        a = x + y  # [1, 64, 112, 112] + [1, 64, 112, 112] = [1, 64, 112, 112]
        b = torch.mul(
            a, z
        )  # [1, 64, 112, 112] * [1, 64, 1, 1] = [1, 64, 112, 112]      a * z channel attn
        c = self.activation(b)
        return c


# ------------------------------#
#   支持重参数化
#             │
#       ┌─────┴─────┐
#       │           │
#    1x1Conv     3x3Conv
#       │           │
#      BN          BN
#       │           │
#       └────add────┘
#             │
# ------------------------------#
class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inference_mode=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=True
            )
        else:
            # self.branch1 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=1))
            # self.branch2 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=3))
            self.branch1 = Conv2dBN(in_channels, out_channels, kernel_size=1)
            self.branch2 = Conv2dBN(in_channels, out_channels, kernel_size=3)

    def forward(self, inputs):
        if self.inference_mode:
            c = self.reparam_conv(inputs)
        else:
            a = self.branch1(inputs)
            b = self.branch2(inputs)
            c = a + b
        return c

    def reparameterize(self):
        self.reparam_conv = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=True
        )
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for param in self.parameters():
            param.detach_()
        self.__delattr__("branch1")
        self.__delattr__("branch2")

        self.inference_mode = True

    def _get_kernel_bias(self) -> tuple[torch.Tensor, torch.TensorType]:
        # 1x1Conv BN
        branch1_kernel, branch1_bias = self._fuse_bn_tensor(self.branch1)
        # 填充一圈0
        branch1_kernel = torch.nn.functional.pad(branch1_kernel, [1, 1, 1, 1])
        # 3X3Conv BN
        branch2_kernel, branch2_bias = self._fuse_bn_tensor(self.branch2)

        kernel_final = branch1_kernel + branch2_kernel
        bias_final = branch1_bias + branch2_bias
        # print(kernel_final.size())  # [out_channels, in_channels, height, width]
        # print(bias_final.size())    # [out_channels]
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


# -----------------------------------#
#             │
#            BN
#             │
#             ├────────────┐
#             │            │
#             │    AdaptiveAvgPool2d
#             │            │
#             │         1x1Conv
#             │            │
#             │         Sigmoid
#             │            │
#            mul───────────┘
#             │
# -----------------------------------#
class SSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(SSEBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)

        self.norm = nn.BatchNorm2d(self.in_channels)

    def forward(self, inputs):
        bn = self.norm(inputs)  # [B, 96, 56, 56] -> [B, 96, 56, 56]

        x = self.globalAvgPool(bn)  # [B, 96, 56, 56] -> [B, 96, 1, 1]
        x = self.conv(x)  # [B, 96, 1, 1] -> [B, 96, 1, 1]
        x = self.sigmoid(x)

        z = torch.mul(
            bn, x
        )  # [B, 96, 56, 56] * [B, 96, 1, 1] = [B, 96, 56, 56]     bn * x channel attn
        return z


# -----------------------------------------------------#
#   支持重参数化
#                       │
#                 ┌─────┴─────┐
#                 │           │
#              FuseBlock   SSEBlock
#                 │           │
#                 └────add────┘
#                       │
#                      SiLU
#                       │
#
#          左侧FuseBlock,右侧是SSEBlock
#                       │
#              ┌────────┴────────┐
#              │                 │
#              │                BN
#              │                 │
#        ┌─────┴─────┐           ├────────────┐
#        │           │           │            │
#      1x1Conv     3x3Conv       │    AdaptiveAvgPool2d
#        │           │           │            │
#       BN          BN           │         1x1Conv
#        │           │           │            │
#        └────add────┘           │         Sigmoid
#              │                 │            │
#              │                mul───────────┘
#              │                 │
#              └───────add───────┘
#                       │
#                      SiLU
#                       │
# -----------------------------------------------------#
class Stream(nn.Module):
    def __init__(self, in_channels, out_channels, inference_mode=False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse = FuseBlock(
            self.in_channels, self.out_channels, inference_mode=inference_mode
        )
        self.sse = SSEBlock(self.in_channels, self.out_channels)
        self.activation = activation_func("silu")

    def forward(self, inputs):
        a = self.fuse(inputs)  # [B, 96, 56, 56] -> [B, 96, 56, 56]
        b = self.sse(inputs)  # [B, 96, 56, 56] -> [B, 96, 56, 56]
        c = a + b  # [B, 96, 56, 56] + [B, 96, 56, 56] = [B, 96, 56, 56]

        d = self.activation(c)
        return d


# ------------------------------------------#
#       │                         │
#      BN                        BN
#       │                         │
#       └───────────cat───────────┘
#                    │
#                 shuffle
#                    │
#       ┌────────────┼────────────┐
#       │            │            │
#   2x2Avgpool       │     AdaptiveAvgPool
#       │            │            │
#    1x1Conv      3x3Conv      1x1Conv
#      g=2        s=2,g=2        g=2
#       │            │            │
#      BN           BN         Sigmoid
#       │            │            │
#       └───────────add           │
#                    │            │
#                   mul───────────┘
#                    │
#                   SiLU
#                    │
# ------------------------------------------#
class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Fusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.network_in_channels = 2 * self.in_channels

        self.avgpool = nn.AvgPool2d(2)

        # self.conv1 = conv_bn(self.network_in_channels, self.out_channels, conv_sampler(kernel_size=1, groups=2))
        # self.conv2 = conv_bn(self.network_in_channels, self.out_channels, conv_sampler(kernel_size=3, stride=2, groups=2))
        # self.conv3 = conv_van(self.network_in_channels, self.out_channels, conv_sampler(kernel_size=1, groups=2))

        self.conv1 = Conv2dBN(
            self.network_in_channels, out_channels, kernel_size=1, groups=2
        )
        self.conv2 = Conv2dBN(
            self.network_in_channels, out_channels, kernel_size=3, stride=2, groups=2
        )
        self.conv3 = nn.Conv2d(
            self.network_in_channels, out_channels, 1, groups=2, bias=False
        )

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.activation = activation_func("silu")
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(self.in_channels)

    def forward(self, input1, input2):
        # 先对2个输入做BN,然后在channel拼接
        a_ = torch.cat(
            [self.bn(input1), self.bn(input2)], dim=1
        )  # [B, 192, 28, 28] cat [B, 192, 28, 28] = [B, 384, 28, 28]

        # shuffle
        idx = torch.randperm(a_.nelement())  # 整数随机排列 无法导出onnx
        a = a_.view(-1)[idx]  # [B, 384, 28, 28] -> [B*384*28*28]
        a = a.view(a_.size())  # [B*384*28*28] -> [B, 384, 28, 28]

        # 分支1
        x = self.avgpool(a)  # [B, 384, 28, 28] -> [B, 384, 14, 14]
        x = self.conv1(x)  # [B, 384, 14, 14] -> [B, 384, 14, 14]

        # 分支2
        y = self.conv2(a)  # [B, 384, 28, 28] -> [B, 384, 14, 14]

        # 分支3
        z = self.globalAvgPool(a)  # [B, 384, 28, 28] -> [B, 384, 1, 1]
        z = self.conv3(z)  # [B, 384, 1, 1] -> [B, 384, 1, 1]
        z = self.sigmoid(z)

        a = x + y  # [B, 384, 14, 14] + [B, 384, 14, 14] = [B, 384, 14, 14]

        b = torch.mul(
            a, z
        )  # [B, 384, 14, 14] * [B, 384, 1, 1] = [B, 384, 14, 14]  a * z channel attn
        c = self.activation(b)
        return c


class ParNetEncoder(nn.Module):
    def __init__(
        self, in_channels, block_size, depth=[4, 5, 5], inference_mode=False
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.block_size = block_size
        self.depth = depth
        self.d1 = Downsample(self.in_channels, self.block_size[0])
        self.d2 = Downsample(self.block_size[0], self.block_size[1])
        self.d3 = Downsample(self.block_size[1], self.block_size[2])
        self.d4 = Downsample(self.block_size[2], self.block_size[3])
        self.d5 = Downsample(self.block_size[3], self.block_size[4])

        # 4层
        self.stream1 = nn.Sequential(
            *[
                Stream(
                    self.block_size[1],
                    self.block_size[1],
                    inference_mode=inference_mode,
                )
                for _ in range(self.depth[0])
            ]
        )

        self.stream1_downsample = Downsample(self.block_size[1], self.block_size[2])

        # 5层
        self.stream2 = nn.Sequential(
            *[
                Stream(
                    self.block_size[2],
                    self.block_size[2],
                    inference_mode=inference_mode,
                )
                for _ in range(self.depth[1])
            ]
        )

        # 5层
        self.stream3 = nn.Sequential(
            *[
                Stream(
                    self.block_size[3],
                    self.block_size[3],
                    inference_mode=inference_mode,
                )
                for _ in range(self.depth[2])
            ]
        )

        # Fusion 会拼接2个输入并下采样
        self.stream2_fusion = Fusion(self.block_size[2], self.block_size[3])
        self.stream3_fusion = Fusion(self.block_size[3], self.block_size[3])

    # ----------------------------------------------------#
    #                                           │ 224
    #                                          d1
    #                                           │ 112
    #           ┌──────────────────────────────d2
    #           │ 56                            │ 56
    #        stream1             ┌─────────────d3
    #           │ 56             │ 28           │ 28
    #   stream1_downsample    stream2          d4
    #           │ 28             │ 28           │ 14
    #           └──────────stream2_fusion    stream3
    #                            │ 14           │ 14
    #                            └────────stream3_fusion
    #                                           │ 7
    #                                           d5
    #                                           │ 4
    # ----------------------------------------------------#
    def forward(self, inputs):
        x = self.d1(inputs)  # [B, 3, 224, 224] -> [B, 64, 112, 112]
        x = self.d2(x)  # [B, 64, 112, 112] -> [B, 96, 56, 56]

        y = self.stream1(x)  # [B, 96, 56, 56] -> [B, 96, 56, 56]
        y = self.stream1_downsample(y)  # [B, 96, 56, 56] -> [B, 192, 28, 28]

        x = self.d3(x)  # [B, 96, 56, 56] -> [B, 192, 28, 28]

        z = self.stream2(x)  # [B, 192, 28, 28] -> [B, 192, 28, 28]
        z = self.stream2_fusion(
            y, z
        )  # [B, 192, 28, 28] and [B, 192, 28, 28] -> [B, 384, 14, 14]

        x = self.d4(x)  # [B, 192, 28, 28] -> [B, 384, 14, 14]

        a = self.stream3(x)  # [B, 384, 14, 14] -> [B, 384, 14, 14]
        b = self.stream3_fusion(
            z, a
        )  # [B, 384, 14, 14] and [B, 384, 14, 14] = [B, 768, 7, 7]

        x = self.d5(b)  # [B, 768, 7, 7] -> [B, 1280, 4, 4]
        return x


class ParNetDecoder(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avg(x)  # [B, 1280, 4, 4] -> [B, 1280, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 1280, 1, 1] -> [B, 1280]
        x = self.decoder(x)  # [B, 1280] -> [B, num_classes]
        # return self.softmax(x)
        return x


class ParNet(nn.Module):
    def __init__(
        self,
        in_channels,
        n_classes,
        block_size=[64, 128, 256, 512, 2048],
        depth=[4, 5, 5],
    ) -> None:
        super().__init__()
        self.encoder = ParNetEncoder(in_channels, block_size, depth)
        self.decoder = ParNetDecoder(block_size[-1], n_classes)

    def forward(self, inputs):
        x = self.encoder(inputs)  # [B, 3, 224, 224] -> [B, 1280, 4, 4]
        x = self.decoder(x)  # [B, 1280, 4, 4] -> [B, num_classes]

        return x


def parnet_sm(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_size=[64, 96, 192, 384, 1280])


def parnet_md(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_size=[64, 128, 256, 512, 2048])


def parnet_l(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_size=[64, 160, 320, 640, 2560])


def parnet_xl(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_size=[64, 200, 400, 800, 3200])


def reparameterize_model(model: nn.Module) -> nn.Module:
    import copy

    model_ = copy.deepcopy(model)
    # children() 只会遍历模型的子层
    # modules()  迭代遍历模型的所有子层,所有子层即指nn.Module子类
    for module in model_.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model_


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = parnet_sm(3, 1000)
    model = reparameterize_model(model)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 1000]

    # 查看结构
    if False:
        onnx_path = "parnet_sm_fuse.onnx"
        torch.onnx.export(
            model,
            x,
            onnx_path,
            input_names=["images"],
            output_names=["classes"],
        )
        import onnx
        from onnxsim import simplify

        # 载入onnx模型
        model_ = onnx.load(onnx_path)

        # 简化模型
        model_simple, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simple, onnx_path)
        print("finished exporting " + onnx_path)
