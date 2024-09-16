# https://github.com/pengzhiliang/Conformer/blob/main/conformer.py
# https://arxiv.org/abs/2105.03889

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, trunc_normal_


# -----------------------------------------#
#   [..., C] -> [..., n*C] -> [..., C]
# -----------------------------------------#
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # mix channel
        x = self.fc1(x)  # [B, 197, 384] -> [B, 197, 1536]
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # [B, 197, 1536] -> [B, 197, 384]
        x = self.drop(x)
        return x


# -------------#
#   自注意力
# -------------#
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        # 每个head的dim维度,每个head均分dim
        head_dim = dim // num_heads
        # -----------------------------------------#
        #   分母的 \sqrt d
        #   如果传入了qk_scale就算,不然就计算: 根号下d_k
        # -----------------------------------------#
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        # -----------------------------------------#
        #   全连接层,直接使用一个全连接层实现求qkv,不使用3个,这样输出长度是3,和用3个一样
        # -----------------------------------------#
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 全连接层,处理拼接后的多头自注意力的输出
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # [N, 197, 768] [B, position, channel]
        B, N, C = x.shape

        # -----------------------------------------------------------#
        #   qkv():   [B, 197, 384] -> [B, 197, 3 * 384]  最后维度变为3倍
        #   reshape: [B, 197, 3 * 384]  -> [B, 197, 3, 6, 64]   3指的是q,k,v; 12是heads个数, 64是每个head所占维度数量
        #   permute: [B, 197, 3, 6, 64] -> [3, b, 6, 197, 64]   3代表q,k,v
        # -----------------------------------------------------------#
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # [B, 6, 197, 64]
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        # -----------------------------------------------------------#
        #   针对每个heads的qk进行计算, qk都是4维数据,相乘是最后两个维度
        #   q乘以k的转置,(-2, -1)就是转置, 再除以根号下d_k
        #   [B, position, channel] @ [B, channel, position]
        #   从前面一个的position来看,一行代表一个position,它会乘以后面的每一列,后面的列也是position,就意味着求位置之间的相似度
        #
        #   [a, b] @ [B, a] = [a, a]  ab代表行列
        #   transpose: [B, 6, 197, 64] -> [B, 6, 64, 197]
        #   @:         [B, 6, 197, 64]  @ [B, 6, 64, 197] = [B, 6, 197, 197]
        # -----------------------------------------------------------#
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # -----------------------------------------------------------#
        #   softmax处理, -1指的是在矩阵中跨列计算,将每一行的总和都设置为1
        #   ex: softmax([[1, 4],[5, 5.]]) = [[0.0474, 0.9526],[0.5000, 0.5000]]
        #   [B, 12, 197, 197] -> [B, 12, 197, 197]
        # -----------------------------------------------------------#
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # -----------------------------------------------------------#
        #   针对每一个v的权重进行矩阵相乘,
        #   @:          [B, 6, 197, 197] @ [B, 6, 197, 64] = [B, 6, 197, 64]  [197, 197]每行经过softmax和[192, 646]的每列相乘,后面的列代表位置,
        #   transpose:  [B, 6, 197, 64] -> [B, 197, 6, 64]
        #   reshape:    [B, 197, 6, 64] -> [B, 197, 384]
        # -----------------------------------------------------------#
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        # 全连接层,处理拼接后的多头自注意力的输出 [B, 197, 384] -> [B, 197, 384]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -----------------------#
#   ENcoder Block
#   Attention + Mlp
# -----------------------#
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(
            self.attn(self.norm1(x))
        )  # [B, 197, 384] -> [B, 197, 384]
        x = x + self.drop_path(
            self.mlp(self.norm2(x))
        )  # [B, 197, 384] -> [B, 197, 384]  mix channel
        return x


# ----------------------------------------------------------------#
#   残差块, 1x1Conv降低通道, 3x3Conv提取特征, 1x1Conv还原通道 残差
# ----------------------------------------------------------------#
class ConvBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        outplanes,
        stride=1,
        res_conv=False,
        act_layer=nn.ReLU,
        groups=1,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
        drop_block=None,
        drop_path=None,
    ):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(
            inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            med_planes,
            med_planes,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(
            med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = norm_layer(outplanes)
        # 残差连接后的激活函数
        self.act3 = act_layer(inplace=True)

        # 残差连接调整通道和步长
        if res_conv:
            self.residual_conv = nn.Conv2d(
                inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = (
            self.conv2(x) if x_t is None else self.conv2(x + x_t)
        )  # 如果有x_t就相加 x_t是Transformer提取的输出,形状和conv2的输入输出相同
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        # 残差连接调整通道和步长
        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        # 残差连接后的激活函数
        x = self.act3(x)

        if return_x_2:
            return x, x2  # x2是3x3Conv的返回值
        else:
            return x


# ------------------------------#
#   融合卷积到Transformer中
# ------------------------------#
class FCUDown(nn.Module):
    """CNN feature maps -> Transformer patch embeddings"""

    def __init__(
        self,
        inplanes,
        outplanes,
        dw_stride,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(
            inplanes, outplanes, kernel_size=1, stride=1, padding=0
        )
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        """
        Args:
            x (Tensor): 残差块中3x3Conv的返回值 ex: [B, 16, 56, 56]
            x_t (Tensor): Transformer结构 ex: [B, 197, 384]

        Returns:
            Tensor: [B, 197, 384]
        """
        x = self.conv_project(x)  # [B, 16, 56, 56] -> [B, 384, 56, 56]

        x = (
            self.sample_pooling(x).flatten(2).transpose(1, 2)
        )  # [B, 384, 56, 56] -> [B, 384, 16, 16] -> [B, 384, 196] -> [B, 196, 384]
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat(
            [x_t[:, 0][:, None, :], x], dim=1
        )  # [B, 1, 384] cat [B, 1, 384] = [B, 197, 384]

        return x


# ------------------------------#
#   从Transformer中提取卷积块
# ------------------------------#
class FCUUp(nn.Module):
    """Transformer patch embeddings -> CNN feature maps"""

    def __init__(
        self,
        inplanes,
        outplanes,
        up_stride,
        act_layer=nn.ReLU,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
    ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(
            inplanes, outplanes, kernel_size=1, stride=1, padding=0
        )
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        """
        Args:
            x (Tensor): 卷积块和TransformerBlock的融合结构  ex: [B, 197, 384]
            H (int): 卷积图片原始高 56
            W (int): 卷积图片原始宽 56

        Returns:
            Tensor: 提取出的卷积结构  ex: [B, 16, 56, 56]
        """
        B, _, C = x.shape
        # [B, 197, 384] -> [B, 196, 384] -> [B, 384, 196] -> [B, 384, 14, 14]
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)  # x[:, 1:] 忽略分类层
        x_r = self.act(
            self.bn(self.conv_project(x_r))
        )  # [B, 384, 14, 14] -> [B, 16, 14, 14]

        return F.interpolate(
            x_r, size=(H * self.up_stride, W * self.up_stride)
        )  # [B, 16, 14, 14] -> [B, 16, 64, 64]


class Med_ConvBlock(nn.Module):
    """special case for Convblock with down sampling,"""

    def __init__(
        self,
        inplanes,
        act_layer=nn.ReLU,
        groups=1,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
        drop_block=None,
        drop_path=None,
    ):
        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(
            inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            med_planes,
            med_planes,
            kernel_size=3,
            stride=1,
            groups=groups,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(
            med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


# ------------------------------#
#   从Transformer中提取卷积块
# ------------------------------#
class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(
        self,
        inplanes,
        outplanes,
        res_conv,
        stride,
        dw_stride,
        embed_dim,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        last_fusion=False,
        num_med_block=0,
        groups=1,
    ):
        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(
            inplanes=inplanes,
            outplanes=outplanes,
            res_conv=res_conv,
            stride=stride,
            groups=groups,
        )

        # 最后的卷积会把 [B, 256, 14, 14] -> [B, 256, 7, 7]
        if last_fusion:
            self.fusion_block = ConvBlock(
                inplanes=outplanes,
                outplanes=outplanes,
                stride=2,
                res_conv=True,
                groups=groups,
            )
        else:
            self.fusion_block = ConvBlock(
                inplanes=outplanes, outplanes=outplanes, groups=groups
            )

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        # ------------------------------#
        #   融合残差块中3x3Conv的输出与Trans的输入
        # ------------------------------#
        self.squeeze_block = FCUDown(
            inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride
        )
        # ------------------------------#
        #   从融合的Transformer结果中取出卷积数据
        # ------------------------------#
        self.expand_block = FCUUp(
            inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride
        )

        self.trans_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
        )

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        """
        Args:
            x (Tensor): 残差块中3x3Conv的返回值 ex: [B, 16, 56, 56]
            x_t (Tensor): Transformer结构 ex: [B, 197, 384]

        Returns:
            Tuple(Tensor): [B, 64, 56, 56], [B, 197, 384]
        """
        # ------------------#
        #   残差部分
        # ------------------#
        x, x2 = self.cnn_block(
            x
        )  # x: [B, 64, 56, 56] -> [B, 64, 56, 56] & [B, 16, 56, 56] x2是3x3Conv的返回值

        _, _, H, W = x2.shape  # 56, 56
        # ------------------------------#
        #   融合残差块中3x3Conv的输出与Transformer的输入
        # ------------------------------#
        x_st = self.squeeze_block(
            x2, x_t
        )  # [B, 16, 56, 56] with [B, 197, 384] -> [B, 197, 384]

        # ------------------#
        #   Transformer部分
        # ------------------#
        x_t = self.trans_block(
            x_st + x_t
        )  # [B, 197, 384] + [B, 197, 384] -> [B, 197, 384]    a TransformerBlock

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        # ------------------------------#
        #   从融合的Transformer结果中取出卷积数据
        # ------------------------------#
        x_t_r = self.expand_block(
            x_t, H // self.dw_stride, W // self.dw_stride
        )  # [B, 197, 384] -> [B, 16, 56, 56]
        x = self.fusion_block(
            x, x_t_r, return_x_2=False
        )  # [B, 64, 56, 56] with [B, 16, 56, 56] -> [B, 64, 56, 56]

        return x, x_t


class Conformer(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        base_channel=64,
        channel_ratio=4,
        num_med_block=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # --------------------#
        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        # --------------------#
        self.conv1 = nn.Conv2d(
            in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # [B, 3, 224, 224] -> [B, 65, 112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )  # [B, 64, 112, 112] -> [B, 64, 56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(
            inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1
        )  # [B, 64, 56, 56] -> [B, 64, 56, 56]
        self.trans_patch_conv = nn.Conv2d(
            64,
            embed_dim,
            kernel_size=trans_dw_stride,
            stride=trans_dw_stride,
            padding=0,
        )  # [B, 64, 56, 56] -> [B, 384, 14, 14] -> [B, 384, 196] -> [B, 196, 384]
        self.trans_1 = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,  # [B, 197, 384] -> [B, 197, 384]
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=self.trans_dpr[0],
        )

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module(
                "conv_trans_" + str(i),
                ConvTransBlock(
                    stage_1_channel,
                    stage_1_channel,
                    False,
                    1,
                    dw_stride=trans_dw_stride,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=self.trans_dpr[i - 1],
                    num_med_block=num_med_block,
                ),
            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module(
                "conv_trans_" + str(i),
                ConvTransBlock(
                    in_channel,
                    stage_2_channel,
                    res_conv,
                    s,
                    dw_stride=trans_dw_stride // 2,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=self.trans_dpr[i - 1],
                    num_med_block=num_med_block,
                ),
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module(
                "conv_trans_" + str(i),
                ConvTransBlock(
                    in_channel,
                    stage_3_channel,
                    res_conv,
                    s,
                    dw_stride=trans_dw_stride // 4,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=self.trans_dpr[i - 1],
                    num_med_block=num_med_block,
                    last_fusion=last_fusion,
                ),
            )
        self.fin_stage = fin_stage

        # --------------------#
        #   Classifier head
        # --------------------#
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        B = x.shape[0]  # [N, 3, 224, 224]

        # pdb.set_trace()
        # -----------------------------#
        #   stem stage
        # -----------------------------#
        x_base = self.maxpool(
            self.act1(self.bn1(self.conv1(x)))
        )  # [B, 3, 224, 224] -> [B, 64, 112, 112] -> [B, 64, 56, 56]

        # -----------------------------#
        #   1 stage
        #   1 stage conv
        # -----------------------------#
        x = self.conv_1(x_base, return_x_2=False)  # [B, 64, 56, 56] -> [B, 64, 56, 56]
        # -----------------------------#
        #   1 stage transformer
        # -----------------------------#
        x_t = (
            self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        )  # [B, 64, 56, 56] -> [B, 384, 14, 14] -> [B, 384, 196] -> [B, 196, 384]
        # vit的分类头
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [1, 1, 384] -> [B, 1, 384]
        x_t = torch.cat(
            [cls_tokens, x_t], dim=1
        )  # [B, 1, 384] cat [B, 196, 384] -> [B, 197, 384]
        x_t = self.trans_1(x_t)  # [B, 197, 384] -> [B, 197, 384]  a TransformerBlock

        # -----------------------------#
        #   2 ~ final
        #   放入卷积和Transformer的输入
        #   返回卷积和Transformer的结果
        # -----------------------------#
        for i in range(
            2, self.fin_stage
        ):  # [B, 64, 56, 56] -> [B, 128, 28, 28] -> [B, 256, 14, 14] -> [B, 256, 7, 7]
            x, x_t = eval("self.conv_trans_" + str(i))(
                x, x_t
            )  # [B, 197, 384] -> [B, 197, 384]

        # -----------------------------#
        #   conv classification
        # -----------------------------#
        x_p = self.pooling(x).flatten(1)  # [B, 256, 7, 7] -> [B, 256, 1, 1] -> [B, 256]
        conv_cls = self.conv_cls_head(x_p)  # [B, 256] -> [B, num_classes]

        # -----------------------------#
        #   trans classification
        # -----------------------------#
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(
            x_t[:, 0]
        )  # [B, 197, 384] 取出 [B, 384] -> [B, num_classes]

        return [conv_cls, tran_cls]  # [[B, num_classes], [B, num_classes]]


def Conformer_tiny_patch16(pretrained=False, **kwargs):
    model = Conformer(
        patch_size=16,
        channel_ratio=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    if pretrained:
        raise NotImplementedError
    return model


def Conformer_small_patch16(pretrained=False, **kwargs):
    model = Conformer(
        patch_size=16,
        channel_ratio=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    if pretrained:
        raise NotImplementedError
    return model


def Conformer_small_patch32(pretrained=False, **kwargs):
    model = Conformer(
        patch_size=32,
        channel_ratio=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    if pretrained:
        raise NotImplementedError
    return model


def Conformer_base_patch16(pretrained=False, **kwargs):
    model = Conformer(
        patch_size=16,
        channel_ratio=6,
        embed_dim=576,
        depth=12,
        num_heads=9,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    if pretrained:
        raise NotImplementedError
    return model


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = Conformer_tiny_patch16(pretrained=False).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y[0].size(), y[1].size())  # [1, 1000] [1, 1000]

    # 查看结构
    if False:
        onnx_path = "Conformer_tiny_patch16.onnx"
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
