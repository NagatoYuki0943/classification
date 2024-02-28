# https://github.com/apple/ml-mobileone/blob/main/mobileone.py
# 使用了重参数化
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from typing import Optional, List, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileOne', 'mobileone', 'reparameterize_model']


class SEBlock(nn.Module):
    """ Squeeze and Excite module.
        Pytorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self,
                 in_channels: int,
                 rd_ratio: float = 0.0625) -> None:
        """ Construct a Squeeze and Excite Module.
        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(in_channels=in_channels,
                                out_channels=int(in_channels * rd_ratio),
                                kernel_size=1,
                                stride=1,
                                bias=True)
        self.expand = nn.Conv2d(in_channels=int(in_channels * rd_ratio),
                                out_channels=in_channels,
                                kernel_size=1,
                                stride=1,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])    # [B, C, H, W] -> [B, C, 1, 1]
        x = self.reduce(x)                              # [B, C, 1, 1] -> [B, C*rd_ratio, 1, 1]
        x = F.relu(x)
        x = self.expand(x)                              # [B, C*rd_ratio, 1, 1] -> [B, C, 1, 1]
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)                         # [B, C, 1, 1] -> [B, C, 1, 1]
        return inputs * x                               # [B, C, H, W] * [B, C, 1, 1] = [B, C, H, W]


#----------------------------------------------------------------------------------------#
#   每个stage都使用 MobileOne block(stage0和stage4只有开始的MobileOne_block_1)
#                   in					rbr_scale和rbr_conv都为conv_bn
#                   │                   以下全部rbr_conv在s0中有4个,都是并行相加,其余模型只有1个
#                   ├──────────┐
#                   │          │
#               rbr_scale   rbr_conv    下采样层 = 1x1Conv + 3x3Conv s=2
#                   │          │        MobileOne_block_1
#                  add─────────┘
#                   │
#                  se
#                   │
#                  act
#                   │
#
#                   │
#                rbr_conv    		    通道变换层rbr_conv = 1x1Conv * n   s0中n=4,其余n=1,就是1个1x1Conv
#                   │                   MobileOne_block_2
#                  se
#                   │
#                  act
#                   │
#
#                   │
#         ┌─────────┼──────────┐
#         │         │          │
#        bn     rbr_scale   rbr_conv    DWConv = bn + 1x1Conv + 3x3Conv ─ ─ ─ ─ ─ ─ ─┐
#         │         │          │        重参数化变为3x3Conv
#         └────────add─────────┘		MobileOne_block_3 5 ...		                 │
#                   │
#                  se 																 │
#                   │
#                  act																 │
#                   │																每个stage会重复n次DW+PW,stage0和stage4重复0次
#																					 │
#                   │
#         ┌─────────┴──────────┐													 │
#         │                    │
#        bn                 rbr_conv    PWConv = bn + 1x1Conv ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
#         │                    │        重参数化变为1x1Conv
#         └────────add─────────┘        MobileOne_block_4 6 ...
#                   │
#                  se
#                   │
#                  act
#                   │
#----------------------------------------------------------------------------------------#
class MobileOneBlock(nn.Module):
    """ MobileOne building block.
        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        For more details, please refer to our paper:
        `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct a MobileOneBlock module.
        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection    BN分支 只有输入输出通道相等且stride=1才使用
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            # Re-parameterizable conv branches      多分支卷积,n个3x3Conv或者1x1Conv
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch       1*1缩放conv分支,
            self.rbr_scale = None
            if kernel_size > 1:                     # 如果其余kernel_size=1,就不需要这个1x1Conv了
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output BN分支
        identity_out = 0
        if self.rbr_skip is not None:   # BN
            identity_out = self.rbr_skip(x)

        # Scale branch output 1*1缩放conv分支
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches    多分支卷积
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83
        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0

        # 1*1缩放conv分支
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # BN分支
        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # 多分支卷积
        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        # 循环获取rbr_conv中的卷积的权重和bias
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        # print(kernel_final.size())    # [out_channels, in_channels, height, width]
        # print(bias_final.size())      # [out_channels]
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95
        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel       = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var  = branch.bn.running_var
            gamma        = branch.bn.weight
            beta         = branch.bn.bias
            eps          = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                # 对角线为1,其余位置为0
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,  # in_channel, out_channel, k, k
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.
        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class MobileOne(nn.Module):
    """ MobileOne Model
        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 num_blocks_per_stage: List[int] = [2, 8, 10, 1], # 每个stage重复block次数(下采样层只有一个MobileOneBlock,非下采样层有2个MobileOneBlock,分别为DWCOnv和PWConv)
                 num_classes: int = 1000,
                 width_multipliers: Optional[List[float]] = None,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1) -> None:
        """ Construct MobileOne model.
        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), num_classes)

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> nn.Sequential:
        """ Build a stage of MobileOne model.
        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)  # 第一次步长为2
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        x = self.stage0(x)          # [B, 3, 224, 224] -> [B, 64, 112, 112]
        x = self.stage1(x)          # [B,64, 112, 112] -> [B, 96, 56, 56]
        x = self.stage2(x)          # [B, 96, 56, 56]  -> [B, 192, 28, 28]
        x = self.stage3(x)          # [B, 192, 28, 28] -> [B, 512, 14, 14]
        x = self.stage4(x)          # [B, 512, 14, 14] -> [B, 1280, 7, 7]
        x = self.gap(x)             # [B, 1280, 7, 7]  -> [B, 1280, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 1280, 1, 1]  -> [B, 1280]
        x = self.linear(x)          # [B, 1280] -> [B, num_classes]
        return x


PARAMS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},
}


def mobileone(num_classes: int = 1000, inference_mode: bool = False,
              variant: str = "s0") -> nn.Module:
    """Get MobileOne model.
    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model. """
    variant_params = PARAMS[variant]
    return MobileOne(num_classes=num_classes, inference_mode=inference_mode, **variant_params)


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.
    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    # children() 只会遍历模型的子层
    # modules()  迭代遍历模型的所有子层,所有子层即指nn.Module子类
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = mobileone(num_classes=1000, inference_mode=False, variant="s1").to(device)
    model = reparameterize_model(model)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 1000]

    # 查看结构
    if False:
        onnx_path = 'mobileone_s1.onnx'
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
