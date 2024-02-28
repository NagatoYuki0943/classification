""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

understanding the shift window: https://github.com/microsoft/Swin-Transformer/issues/38

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


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
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


#----------------------------------------------------------------#
#   通过image mask划分窗口
#----------------------------------------------------------------#
def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: [B, H, W, C]   ex:[B, 56, 56, 96]
        window_size (int): window size(M) 7

    Returns:
        windows: [B*num_windows, window_size_h, window_size_w, C]   [b*64, 7, 7, 96]
    """
    B, H, W, C = x.shape
    #   [B, 56, 56, 96] -> [B, 8, 7, 8, 7, 96]                  [B, Hn*Mh, Wn*Mw, c] -> [B, Hn, Mh, Wn, Mw, c]
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    #----------------------------------------------------------------#
    #   permute: [B, 8, 7, 8, 7, 96] -> [B, 8, 8, 7, 7, 96]     [B, Hn, Mh, Wn, Mw, c] -> [B, Hn, Wn, Mh, Mw, c]
    #   view:    [B, 8, 8, 7, 7, 96] -> [b*64, 7, 7, 96]        [B, Hn, Wn, Mh, Mw, c] -> [b*Hn*Wn, Mh, Mw, c]
    #----------------------------------------------------------------#
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


#----------------------------------------------------------------#
#   将划分的窗口还原
#----------------------------------------------------------------#
def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: [B*num_windows, window_size_h, window_size_w, C]   ex:[b*64, 7, 7, 96]
        window_size (int): Window size(M)       7
        H (int): Height of image    分隔之前的H  56
        W (int): Width of image     分隔之间的W  56

    Returns:
        x: [B, H, W, C]   ex:[B, 56, 56, 96]
    """
    #----------------------------------------------------------------#
    #   获取batch数量, batch / windows数量   windows数量 = 原始高*原始宽/7/7
    #   b*64 / (56*56/7/7) = b
    #----------------------------------------------------------------#
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    #----------------------------------------------------------------#
    #   [b*64, 7, 7, 96] -> [B, 8, 8, 7, 7, 96]             [b*Hn*Wn, Mh, Mw, c] -> [B, Hn, Wn, Mh, Mw, c]
    #----------------------------------------------------------------#
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    #----------------------------------------------------------------#
    #   permute: [B, 8, 8, 7, 7, 96] -> [B, 8, 7, 8, 7, 96] [B, Hn, Wn, Mh, Mw, c] -> [B, Hn, Mh, Wn, Mw, c]
    #   view:    [B, 8, 7, 8, 7, 96] -> [B, 56, 56, 96]     [B, Hn, Mh, Wn, Mw, c] -> [B, Hn*Mh, Wn*Mw, c]
    #----------------------------------------------------------------#
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#------------------------------------------------------------------#
#   将图片划分为没有重叠的patch Patch Partition和Linear Embedding
#   直接使用 k=4,s=4,out_channel=96的卷积           out=[B, position, channel]
#   [B, 3, 224, 224] -> [B, 96, 56, 56] -> [B, 96, 56*56] -> [B, 56*56, 96]
#   return: [B, 56*56, 96], 56, 56
#------------------------------------------------------------------#
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        # k=4,s=4,out_channel=96
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # [B, 3, 224, 224] -> [B, 96, 56, 56]
        x = self.proj(x)
        _, _, H, W = x.shape
        # [B, 96, 56, 56] -> [B, 96, 56*56] -> [B, 56*56, 96]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


#--------------------------------------------------------------#
#   宽高减半,通道翻倍
#   通过跳采样宽高减半,通道翻4倍,再通过线性层通道减半
#   [B, 56*56, 96] -> [B, 28*28, 192] -> [B, 14*14, 384] -> [B, 7*7, 768]
#--------------------------------------------------------------#
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        #----------------------------------#
        #   通过卷积线性层通道减半
        #----------------------------------#
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        # 长度必须为宽*高
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # [B, 56*56, 96] -> [B, 56, 56, 96]
        x = x.view(B, H, W, C)

        # padding 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward. 注意padding参数2是从后往前的
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            # 0, 0:     通道
            # 0, W % 2: 宽度 右侧 padding 0
            # 0, H % 2: 高度 下侧 padding 0

        #-----------------------------------------#
        #   宽高跳采样,注意现在通道在最后
        #-----------------------------------------#
        x0 = x[:, 0::2, 0::2, :]  # [B, 28, 28, 96]
        x1 = x[:, 1::2, 0::2, :]  # [B, 28, 28, 96]
        x2 = x[:, 0::2, 1::2, :]  # [B, 28, 28, 96]
        x3 = x[:, 1::2, 1::2, :]  # [B, 28, 28, 96]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, 28, 28, 96*4]

        x = x.view(B, -1, 4 * C)  # [B, 28, 28, 96*4] -> [B, 28*28, 96*4]
        x = self.norm(x)
        x = self.reduction(x)  # [B, 28*28, 96*4] -> [B, 28*28, 96*2]

        return x


#-----------------------------------------------------------#
#   MLP
#   Linear -> GELU -> Dropout -> Linear -> Dropout  两个Dropout共用
#   第一个linear通道翻四倍,第二个Linear通道还原
#   [..., C] -> [..., n*C] -> [..., C]
#-----------------------------------------------------------#
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    in_features:     输入通道数
    hidden_features: 一般是 in_features*4
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 如果传入out_features,就使用,不然就是in_features
        out_features = out_features or in_features
        # 如果传入hidden_features,就使用,不然就是in_features
        hidden_features = hidden_features or in_features

        # 通道变为4倍
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        # 通道还原
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):   # mix channel
        x = self.fc1(x)     # [B, 56*56, 96] -> [B, 56*56, 384]
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)     # [B, 56*56, 384] -> [B, 56*56, 96]
        x = self.drop2(x)
        return x

#----------------------------------------------------------------#
#   W-MSA和S-MSA的Attentio
#   [B, position, channel] -> [B, position, channel] @ [B, channel, position] @ [B, position, channel] = [B, position, channel]
#----------------------------------------------------------------#
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.                                通道数
        window_size (tuple[int]): The height and width of the window.       w-sma sw-msa采用窗口大小 7
        num_heads (int): Number of attention heads.                         堆叠多少次swim block
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads

        # 每个头的通道数
        head_dim = dim // num_heads
        #-------------------------#
        #   multi head中的\sqrt d
        #-------------------------#
        self.scale = head_dim ** -0.5

        #------------------------------------#
        #   相对位置偏置,是一维向量,长度是(2M-1)**2 ,注意每个head都有自己的相对位置偏置
        #   define a parameter table of relative position bias
        #------------------------------------#
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        #------------------------------------#
        #   生成relative position
        #   get pair-wise relative position index for each token inside the window
        #------------------------------------#
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        #------------------------------------#
        #   二维索引变为一维索引
        #------------------------------------#
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        #------------------------------------#
        #   放到模型缓存中
        #------------------------------------#
        self.register_buffer("relative_position_index", relative_position_index)

        #-----------------------------------------#
        #   全连接层,直接使用一个全连接层实现求qkv,不使用3个,这样输出长度是3,和用3个一样
        #-----------------------------------------#
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        #-----------------------------------------#
        #   全连接层,处理拼接后的多头自注意力的输出
        #-----------------------------------------#
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        #------------------------------------#
        #   [b*64, 49, 96] [b*num_windows, Mh*Mw, channel]  Mh,Mw是每个小窗口的宽高
        #------------------------------------#
        B_, N, C = x.shape
        # qkv():   [b*64, 49, 96] -> [b*64, 49, 3 * 96]          最后channnel变为3倍
        # reshape: [b*64, 49, 3 * 96]   -> [b*64, 49, 3, 3, 32]  第一个3指的是q,k,v; 第二个3是heads个数, 最后是每个head所占channnel数量
        # permute: [b*64, 49, 3, 3, 32] -> [3, b*64, 3, 49, 32]  第一个3代表q,k,v
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        #-----------------------------------------------------------#
        #   通过切片拿到q,k,v数据,切片之后第一个维度就没有了,绝了
        #   [b*64, 3, 49, 32]
        #-----------------------------------------------------------#
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        #-----------------------------------------------------------#
        #   针对每个heads的qk进行计算, qk都是4维数据,相乘是最后两个维度
        #   q乘以k再转置,(-2, -1)就是转置, 再除以根号下d_k
        #   [B, position, channel] @ [B, channel, position]
        #   从前面一个的position来看,一行代表一个position,它会乘以后面的每一列,后面的列也是position,就意味着求位置之间的相似度
        #
        #   transpose: [b*64, 3, 49, 32] -> [b*64, 3, 32, 49]
        #   @:         [b*64, 3, 49, 32]  @ [b*64, 3, 32, 49] = [b*64, 3, 49, 49]
        #-----------------------------------------------------------#
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        #-----------------------------------------------------------#
        #   通过relative position取relative parameter
        #   relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]     nH: num_head
        #   [49*49, 3] -> [49, 49, 3]
        #-----------------------------------------------------------#
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [49, 49, 3] -> [, 349, 49]
        #------------------------------------#
        #   Attention(Q,K,V) = softmax(\frac {QK^T} {\sqrt d} + B)V  中加上 B 的过程
        #   [b*64, 3, 49, 49] + [1, 3, 49, 49] = [b*64, 3, 49, 49]
        #------------------------------------#
        attn = attn + relative_position_bias.unsqueeze(0)

        #-----------------------------------------------------------#
        #   softmax处理, -1指的是取每一列,在行上做softmax
        #   ex: softmax([[1, 4],[5, 5.]]) = [[0.0474, 0.9526],[0.5000, 0.5000]]
        #   [b*64, 3, 49, 49] -> [b*64, 3, 49, 49]
        #-----------------------------------------------------------#
        if mask is not None:
            #------------------------------------#
            #   mask: [nW, Mh*Mw, Mh*Mw]
            #------------------------------------#
            nW = mask.shape[0]  # num_windows
            # attn.view: [B, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)   # mask相同区域使用0表示,不同区域使用-100表示,通过softmax之后就很小了
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        #-----------------------------------------------------------#
        # @:         [b*64, 3, 49, 49]  @ [b*64, 3, 49, 32] = [b*64, 3, 49, 32]
        # transpose: [b*64, 3, 49, 32] -> [b*64, 49, 3, 32]
        # reshape:   [b*64, 49, 3, 32] -> [b*64, 49, 96]
        #-----------------------------------------------------------#
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)

        # 全连接层,处理拼接后的多头自注意力的输出 [b*64, 49, 96] -> [b*64, 49, 96]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#----------------------------------------------------------------#
#   W-SMA or SM-WSA 包含了多头自注意力和MLP部分
#   LN -> W-MSA/SW-MSA -> DropPath -> 残差 -> LN -> MLP -> DropPath -> 残差
#----------------------------------------------------------------#
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.                        通道数
        num_heads (int): Number of attention heads.                 堆叠多少次swim block
        window_size (int): Window size.                             w-sma sw-msa采用窗口大小 7
        shift_size (int): Shift size for SW-MSA.                    0 or shift_size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.shift_size  = shift_size
        self.mlp_ratio   = mlp_ratio
        #------------------------------------#
        #   shift size必须在0~window之间
        #------------------------------------#
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        #------------------------------------#
        #   W-MSA/SW-MSA 根据shift_size来判断
        #------------------------------------#
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        #------------------------------------#
        #   Multi-Head和MLP后面的DropPath
        #------------------------------------#
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        #   dim * 4
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop) # drop指的是mlp中使用的drop

    def forward(self, x, attn_mask):
        """
        x:  [B, 56*56, 96]
        """
        # 输入对应的宽高 56, 56
        H, W = self.H, self.W
        B, L, C = x.shape
        #   L = H * W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)  # [B, 56*56, 96] -> [B, 56, 56, 96]

        #------------------------------------#
        #   pad feature maps to multiples of window size
        #   把feature map给pad到window size的整数倍
        #   右下侧padding
        #------------------------------------#
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # 获取padding后的宽高 56,56
        _, Hp, Wp, _ = x.shape

        #------------------------------------#
        #   如果是SW-MSA,移动数据
        #   cyclic shift
        #------------------------------------#
        if self.shift_size > 0:
            #   SW-MSA 通过roll移动
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            # W-MSA 将mask设置为None
            shifted_x = x   # [1, 56, 56, 96]
            attn_mask = None

        #------------------------------------#
        #   划分窗口
        #   partition windows
        #------------------------------------#
        x_windows = window_partition(shifted_x, self.window_size)  # [B, 56, 56, 96] -> [b*64, 7, 7, 96] 64是nW,64个窗口; 7是Mh和Mw,每个小窗口的宽高
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*64, 49, 96]

        #------------------------------------#
        #   W-MSA/SW-MSA [b*64, 49, 96] -> [b*64, 49, 96]
        #------------------------------------#
        attn_windows = self.attn(x_windows, mask=attn_mask)

        #------------------------------------#
        #   将一个个window还原成一个feature map
        #   merge windows
        #------------------------------------#
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [b*64, 49, 96] -> [b*64, 7, 7, 96]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [b*64, 7, 7, 96] -> [B, 56, 56, 96]

        #------------------------------------#
        #   SW-MSA,移动的数据还原
        #   reverse cyclic shift
        #------------------------------------#
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C) # [B, 56, 56, 96] -> [B, 56*56, 96]

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x))) # [B, 56*56, 96] -> [B, 56*56, 96]

        return x


#------------------------------------------------------------------#
#   SwinTransformerBlock + patch_merging
#   这里的stage不包含该stage的patch_merging层，而包含的是下个stage的patch_merging
#   原因是第一个Linear Embedding已经在上面实现了
#------------------------------------------------------------------#
class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.                            通道数
        depth (int): Number of blocks.                                  堆叠多少次swim block
        num_heads (int): Number of attention heads.                     多头注意力模块head个数
        window_size (int): Local window size.                           w-sma sw-msa采用窗口大小 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.    mlp中第一个linear翻倍数
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None    下采样方式 PatchMerging
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2      # SW-MSA偏移像素值,是窗口大小的一半 7 // 2 = 3

        #----------------------------------------------------------------#
        #   build swin block
        #   每个swin block 依次由 W-SMA 和 SM-WSA 构成,如果重复次数为2,两个模型都是一个
        #----------------------------------------------------------------#
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,  # W-SMA or SM-WSA   0,2...为W-SMA
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 下采样,通过跳采样宽高减半,通道翻倍
        if downsample is not None:
            # dim是输入通道数,返回的通道翻倍,宽高减半
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    #----------------------------#
    #   SM-WSA 创建的模板
    #----------------------------#
    def create_mask(self, x, H, W):
        #----------------------------------------------------------------#
        #   calculate attention mask for SW-MSA
        #   保证Hp和Wp是window_size的整数倍
        #----------------------------------------------------------------#
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        #----------------------------------------------------------------#
        #   拥有和feature map一样的通道排列顺序，方便后续window_partition
        #----------------------------------------------------------------#
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]

        # 分块
        h_slices = (slice(0, -self.window_size),                # 假设9x9,分成3份,shift=1,取0~5
                    slice(-self.window_size, -self.shift_size), # 假设9x9,分成3份,shift=1,取6~7
                    slice(-self.shift_size, None))              # 假设9x9,分成3份,shift=1,取8    下面的w相同
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        #----------------------------------------------------------------#
        #   对相同区域设置cnt,先取h,再取w,
        #   第一次取h,遍历w,是[0~5,0~5],然后是[0~5,6~7],最后是[0~5,8]
        #----------------------------------------------------------------#
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        #----------------------------#
        #   通过image mask划分窗口
        #----------------------------#
        mask_windows = window_partition(img_mask, self.window_size)  # [B*num_windows, Mh, Mw, 1] = [所有窗口,窗口高,窗口宽,通道]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [B*num_windows, Mh*Mw] 每个window都按照行展平
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [B*num_windows, 1, Mh*Mw] - [B*num_windows, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        # 放在这里可以解决多尺度问题
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw] [64, 49, 49]

        # 遍历blocks [B, 56*56, 96] -> [B, 56*56, 96]
        for blk in self.blocks:
            # 给block添加宽高属性
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        # 下采样 [B, 56*56, 96] -> [B, 28*28, 192]
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            # 防止是奇数向下取整
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4                               开始的patchsize,4x4的方块
        in_chans (int): Number of input image channels. Default: 3                          输入图片通道
        num_classes (int): Number of classes for classification head. Default: 1000         分类数
        embed_dim (int): Patch embedding dimension. Default: 96                             Linear Embedding之后的通道数 96
        depths (tuple(int)): Depth of each Swin Transformer layer.                          每个stage重复swim block的次数 (2, 2, 6, 2)
        num_heads (tuple(int)): Number of attention heads in different layers.              多头注意力模块head个数 (3, 6, 12, 24)
        window_size (int): Window size. Default: 7                                          w-sma sw-msa采用窗口大小
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4             mlp中第一个linear翻倍数
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True  是否使用qkv偏置
        drop_rate (float): Dropout rate. Default: 0                                         path embedding层后面的Dropout
        attn_drop_rate (float): Attention dropout rate. Default: 0                          多头注意力模块中的Dropout
        drop_path_rate (float): Stochastic depth rate. Default: 0.1                         swim block中的droppath,逐渐递增
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.                 LN
        patch_norm (bool): If True, add normalization after patch embedding. Default: True  patch embedding后的LN
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False  节省内存
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)   # stage个数=4
        self.embed_dim = embed_dim      # Linear Embedding之后的通道数
        self.patch_norm = patch_norm    # patch embedding后的LN
        #----------------------------------------#
        #   stage4输出特征矩阵的channels
        #----------------------------------------#
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1)) # embed_dim * 2 ** 3 = embed_dim * 8
        self.mlp_ratio = mlp_ratio

        #------------------------------------------------------------------#
        #   将图片划分为没有重叠的patch Patch Partition和Linear Embedding
        #   直接使用 k=4,s=4,out_channel=96的卷积
        #   b,3,224,224 -> b,96,56,56 -> b,96,56*56 -> b,56*56,96   b,每个框,每个框的长度
        #   返回 b,56*56,96 高 宽
        #------------------------------------------------------------------#
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # drop path 逐渐递增
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        #------------------------------------------------------------------#
        #   Swim Blocks [B, 56*56, 96] -> [B, 28*28, 192] -> [B, 14*14, 384] -> [B, 7*7, 768]
        #------------------------------------------------------------------#
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，而包含的是下个stage的patch_merging
            # 原因是第一个patch_merging已经在上面实现了
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),  # 通道数
                                depth=depths[i_layer],              # 堆叠多少次swim block
                                num_heads=num_heads[i_layer],       # 多头注意力模块head个数
                                window_size=window_size,            # w-sma sw-msa采用窗口大小
                                mlp_ratio=self.mlp_ratio,           # mlp中第一个linear翻倍数
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,   # 下采样方式,判断是前三个有下采样,最后没有
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, 3, 224, 224] -> [B, 56*56, 96]  [B, position, channel]
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        #----------------------------------------------------#
        #   swin block
        #   [B, 56*56, 96] -> [B, 784, 192] -> [B, 196, 384] -> [B, 49, 768]
        #----------------------------------------------------#
        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)                        # [B, 49, 768] -> [B, 49, 768]
        x = self.avgpool(x.transpose(1, 2))     # [B, 768, 49] -> [B, 768, 1]   宽高均值
        x = torch.flatten(x, 1)                 # [B, 768, 1] -> [B, 768]
        x = self.head(x)                        # [B, 768] -> [B, num_classes]
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=12,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            **kwargs)
    return model


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = swin_small_patch4_window7_224(num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]
