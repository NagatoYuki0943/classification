"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""

from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.hub import load_state_dict_from_url


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    将卷积核个数(输出通道个数)调整为最接近round_nearest的整数倍,就是8的整数倍,对硬件更加友好
    v:          输出通道个数
    divisor:    奇数,必须将ch调整为它的整数倍
    min_value:  最小通道数

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# Conv+BN+Act
class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        block = nn.Sequential()

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias
        )

        block.add_module(name="conv", module=conv_layer)

        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)
            block.add_module(name="norm", module=norm_layer)

        if use_act:
            act_layer = nn.SiLU()
            block.add_module(name="act", module=act_layer)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


#--------------------------------------#
#   倒残差结构
#   残差:   两端channel多,中间channel少
#       降维 --> 升维
#   倒残差: 两端channel少,中间channel多
#       升维 --> 降维
#   1x1 3x3DWConv 1x1
#   最后的1x1Conv没有激活函数
#--------------------------------------#
class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        skip_connection: Optional[bool] = True,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]
        # 计算通道增加数
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        block = nn.Sequential()
        # 利用1x1卷积根据输入进来的通道数进行通道数上升,不扩张就不需要第一个1x1卷积了
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )
        # 进行3x3的DW卷积
        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim   # in_channels = out_channels = groups
            ),
        )
        # 利用1x1卷积进行通道数的调整,没有激活函数
        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,  # 没有激活函数
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        # 步长为1同时通道不变化才相加
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


#---------------------------------------------------#
#   [N, P, C] -> [N, h, P, c] @ [N, h, c, P] @ [N, h, P, c] = [N, P, C]
#   C = h * c
#   这里的N不是Batch,因为将特征图划分为了2x2的patch,对应位置和对应位置做自注意力,所以 N = 4B
#---------------------------------------------------#
class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (float): Attention dropout. Default: 0.0
        bias (bool): Use bias or not. Default: ``True``
    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )
        # 全连接层,直接使用一个全连接层实现求qkv,不使用3个,这样输出长度是3,和用3个一样
        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        # 全连接层,处理拼接后的多头自注意力的输出
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x_q: Tensor) -> Tensor:
        # [N, P, C]
        b_sz, n_patches, in_channels = x_q.shape

        # self-attention
        # [N, P, C] -> [N, P, 3C] -> [N, P, 3, h, c] where C = hc
        qkv = self.qkv_proj(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)

        # [N, P, 3, h, c] -> [N, h, 3, P, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, P, C] -> [N, h, P, C] x 3
        query, key, value = qkv.unbind(2)

        query = query * self.scaling

        # [N, h, P, c] -> [N, h, c, P]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, P, c] x [N, h, c, P] -> [N, h, P, P]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)       # 取每一列,在行上做softmax
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, P] x [N, h, P, c] -> [N, h, P, c]
        out = torch.matmul(attn, value)

        # [N, h, P, c] -> [N, P, h, c] -> [N, P, C]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)

        # [N, P, C] -> [N, P, C]
        out = self.out_proj(out)

        return out


#--------------------#
#   self_attn + mlp
#--------------------#
class TransformerEncoder(nn.Module):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (int) : Number of heads in multi-head attention. Default: 8
        attn_dropout (float): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers. Default: 0.0
    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        # 多头自注意力
        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True
        )
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            attn_unit,
            nn.Dropout(p=dropout)
        )

        # mlp   [N, P, C] -> [N, P, 2C] -> [N, P, C]
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            nn.Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        # multi-head attention
        res = x
        # [N, P, C] -> [N, h, P, c] @ [N, h, c, P] @ [N, h, P, c] = [N, P, C]
        x = self.pre_norm_mha(x)
        x = x + res

        # [N, P, C] -> [N, P, 2C] -> [N, P, C]
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,           # transformer的token长度
        ffn_dim: int,                   # mlp节点个数
        n_transformer_blocks: int = 2,  # 重复堆叠transformer次数
        head_dim: int = 32,             # 每个头的维度
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,               # 划分特征图的大小
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,  # 开始conv的kernel
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        #-----------------------------#
        #   local representation
        #   3x3Conv + 1x1Conv
        #-----------------------------#
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3",
                                  module=ConvLayer(
                                    in_channels=in_channels,
                                    out_channels=in_channels,
                                    kernel_size=conv_ksize,
                                    stride=1
                                ))
        self.local_rep.add_module(name="conv_1x1",
                                  module=ConvLayer(
                                    in_channels=in_channels,
                                    out_channels=transformer_dim,
                                    kernel_size=1,
                                    stride=1,
                                    use_norm=False,
                                    use_act=False
                                ))

        #----------------------------------#
        #   n 次 transformer
        #----------------------------------#
        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        #----------------------------------#
        #   调整transformer的通道
        #   1x1Conv
        #----------------------------------#
        self.conv_proj = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        #----------------------------------#
        #   fusion res+x的输出经过3x3Conv融合
        #   3x3Conv
        #----------------------------------#
        self.fusion = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.patch_h = patch_h  # 2
        self.patch_w = patch_w  # 2
        self.patch_area = self.patch_w * self.patch_h   # 4

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        """将输入的图片划分为2x2的patch，左上，右上，左下，右下的四个位置的数据
        只和对应位置的数据做自注意力，减小计算量到原来的大约1/4

        Args:
            x (Tensor): [B, C, H, W]

        Returns:
            Tuple[Tensor, Dict]: [4B, H*W/4, C], info
        """
        patch_w, patch_h = self.patch_w, self.patch_h       # 2 2
        patch_area = patch_w * patch_h                      # 4
        batch_size, in_channels, orig_h, orig_w = x.shape   # [b, 64, 28, 28]

        # 保证特征图宽高可以被patch整除
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h) # 28 / 2 * 2 = 28
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w) # 28 / 2 * 2 = 28
        # 如果不被整除就采样到制定分辨率
        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        # 宽度和高度各划分出多个patch
        num_patch_w = new_w // patch_w          # n_w=14
        num_patch_h = new_h // patch_h          # n_h=14
        num_patches = num_patch_h * num_patch_w # N=14*14

        # [B, C, H, W] -> [B * C, n_h, h, n_w, w]                                   [B, 64, 28, 28] -> [B*64, 14, 2, 14, 2]
        x = x.reshape(batch_size * in_channels, num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C, n_h, h, n_w, w] -> [B * C, n_h, n_w, h, w]                        [B*64, 14, 2, 14, 2] -> [B*64, 14, 14, 2, 2]
        x = x.transpose(2, 3)
        # [B * C, n_h, n_w, h, w] -> [B, C, N, P] where P = h * w and N = n_h * n_w [B*64, 14, 14, 2, 2] -> [b, 64, 14*14, 2*2]
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]                                              [B, 64, 196, 4] -> [B, 4, 196, 64]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [B * P, N, C]                                             [B, 4, 196, 64] -> [B*4, 196, 64]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        """将划分为2x2patch的数据还原为[B, C, H, W]形式

        Args:
            x (Tensor): [4B, H*W/4, C]
            info_dict (Dict): info

        Returns:
            Tensor: [B, C, H, W]
        """
        n_dim = x.dim() # 3
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [B * P, N, C] --> [B, P, N, C]                       [B*4, 196, 64] -> [B, 4, 196, 64]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()  # [B, 4, 196, 64]
        num_patch_h = info_dict["num_patches_h"]              # 14
        num_patch_w = info_dict["num_patches_w"]              # 14

        # [B, P, N, C] -> [B, C, N, P]                          [B, 4, 196, 64] -> [B, 64, 196, 4]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C, n_h, n_w, h, w]                 [B, 64, 196, 4] -> [B*64, 14, 14, 2, 2]
        x = x.reshape(batch_size * channels, num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C, n_h, n_w, h, w] -> [B*C, n_h, h, n_w, w][B*64, 14, 14, 2, 2] -> [B*64, 14, 2, 14, 2]
        x = x.transpose(2, 3)
        # [B*C, n_h, h, n_w, w] -> [B, C, H, W]                 [B*64, 14, 2, 14, 2] -> [B, 64, 14*2, 14*2]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)

        # 如果经过采样,则还原大小
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        # local representation  3x3Conv + 1x1Conv
        fm = self.local_rep(x)                        # [B, 48, 28, 28] -> [B, 64, 28, 28]

        # convert feature map to patches
        # [B, C, H, W] -> [4B, H*W/4, C]                [B, 64, 28, 28] -> [4B, 196, 64]
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        # [4B, H*W/4, C] -> [4B, H*W/4, C]              [4B, 196, 64] -> [4B, 196, 64]
        patches = self.global_rep(patches)

        # [4B, H*W/4, C] -> [B, C, H, W]                [4B, 196, 64] -> [B, 64, 28, 28]
        fm = self.folding(x=patches, info_dict=info_dict)

        # 调整transformer的通道   1x1Conv
        fm = self.conv_proj(fm)                       # [B, 64, 28, 28] -> [B, 48, 28, 28]

        # fusion res+x的输出经过3x3Conv融合
        fm = self.fusion(torch.cat((res, fm), dim=1)) # [B, 48, 28, 28] cat [B, 48, 28, 28] -> [B, 48, 28, 28]
        return fm


class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()

        image_channels = 3
        out_channels = 16

        # [B, 3, 224, 224] -> [B, 16, 112, 112]
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

        # [B, 16, 112, 112]-> [B, 16, 112, 112]
        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        # [B, 16, 112, 112]-> [B, 24, 56, 56]
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        # [B, 24, 56, 56] -> [B, 48, 28, 28]
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        # [B, 48, 28, 28] -> [B, 64, 14, 14]
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        # [B, 64, 14, 14] -> [B, 80,  7,  7]
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])
        # [B, 80,  7,  7] -> [B, 320, 7,  7] 最终的1x1Conv增加通道,和MobileNet类似
        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )

        # [B, 320, 7,  7] -> [B, num_classes]
        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        # mobilevit or mobilenet
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            # 第一个stride使用配置的,后面全为1
            stride = cfg.get("stride", 1) if i == 0 else 1
            # 全是倒残差
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        # 获取步长
        stride = cfg.get("stride", 1)
        block = []

        # 步长为2使用MVBlock下采样
        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        # 构建mobilevit block
        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)          # [B, 3, 224, 224] -> [B, 16, 112, 112]
        x = self.layer_1(x)         # [B, 16, 112, 112]-> [B, 16, 112, 112]
        x = self.layer_2(x)         # [B, 16, 112, 112]-> [B, 24, 56, 56]

        x = self.layer_3(x)         # [B, 24, 56, 56] -> [B, 48, 28, 28]
        x = self.layer_4(x)         # [B, 48, 28, 28] -> [B, 64, 14, 14]
        x = self.layer_5(x)         # [B, 64, 14, 14] -> [B, 80,  7,  7]
        x = self.conv_1x1_exp(x)    # [B, 80,  7,  7] -> [B, 320, 7,  7]
        x = self.classifier(x)      # [B, 320, 7,  7] -> [B, num_classes]
        return x


def get_config(mode: str = "xxs") -> dict:
    if mode == "xx_small":
        mv2_exp_mult = 2
        config = {
            "layer1": {
                "out_channels": 16,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 24,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 48,
                "transformer_channels": 64,
                "ffn_dim": 128,
                "transformer_blocks": 2,
                "patch_h": 2,  # 8,
                "patch_w": 2,  # 8,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 4,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 80,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    elif mode == "x_small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 48,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 64,
                "transformer_channels": 96,
                "ffn_dim": 192,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 80,
                "transformer_channels": 120,
                "ffn_dim": 240,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    elif mode == "small":
        mv2_exp_mult = 4
        config = {
            "layer1": {
                "out_channels": 32,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 1,
                "stride": 1,
                "block_type": "mv2",
            },
            "layer2": {
                "out_channels": 64,
                "expand_ratio": mv2_exp_mult,
                "num_blocks": 3,
                "stride": 2,
                "block_type": "mv2",
            },
            "layer3": {  # 28x28
                "out_channels": 96,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 2,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer4": {  # 14x14
                "out_channels": 128,
                "transformer_channels": 192,
                "ffn_dim": 384,
                "transformer_blocks": 4,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 160,
                "transformer_channels": 240,
                "ffn_dim": 480,
                "transformer_blocks": 3,
                "patch_h": 2,
                "patch_w": 2,
                "stride": 2,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    else:
        raise NotImplementedError

    for k in ["layer1", "layer2", "layer3", "layer4", "layer5"]:
        config[k].update({"dropout": 0.1, "ffn_dropout": 0.0, "attn_dropout": 0.0})

    return config


model_urls = {
    'xx_small': 'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt',
    'x_small' : 'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt',
    'small'   : 'https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt',
}


def _create_mobilevit(variant, pretrained=False, num_classes=1000):
    config = get_config(variant)
    model = MobileViT(config, num_classes=num_classes)
    if pretrained:
        state_dict_url = model_urls[variant]
        assert state_dict_url != '', 'There is no pretrained weight for this model!'
        state_dict = load_state_dict_from_url(state_dict_url, progress=True)
        model.load_state_dict(state_dict)
    return model


def mobile_vit_xx_small(pretrained=False, num_classes: int = 1000):
    m = _create_mobilevit('xx_small', pretrained, num_classes)
    return m


def mobile_vit_x_small(pretrained=False, num_classes: int = 1000):
    m = _create_mobilevit('x_small', pretrained, num_classes)
    return m


def mobile_vit_small(pretrained=False, num_classes: int = 1000):
    m = _create_mobilevit('small', pretrained, num_classes)
    return m


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 256, 256).to(device)
    model = mobile_vit_xx_small(pretrained=False).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 1000]
