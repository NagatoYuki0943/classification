"""
输入图片大小不能改变

必须使用预训练模型,除非数据量非常大,比如ImageNet21K

original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn


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
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


#----------------------------------------------------------#
#   Patch [B, C, H, W]-> [B, N, C]
#   卷积核大小是 16x16, 步长是16, out_channel=768   out=[B, position, channel]
#   [B, 3, 224, 224] -> [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
#----------------------------------------------------------#
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        # 224 / 16 = 14
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # 14 x 14 = 196
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # [B, 3, 224, 224] -> [B, 768, 14, 14]  k=16 s=16
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # [B, 3, 224, 224] -> [B, 768, 14, 14]
        x = self.proj(x)
        # [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)    # ❗❗❗important❗❗❗
        return x


#----------------------------------------------------------#
#   实现Transformer中的Multi-Head模块
#   输入的数据经过 Wq Wk Wv 生成 query,key和value,然后再均分成不同的head
#   [B, N, C] -> [B, N, C] @ [B, C, N] @ [B, N, C] = [B, N, C]                                单头
#   [B, N, C] -> [B, h, N, c] @ [B, h, c, N] @ [B, h, N, c] = [B, h, N, c] -> [B, N, C]       多头
#   [B, 197, 768] -> [B, 197, 768] @ [B, 768, 197] @ [B, 197, 768] = [B, 197, 768]            单头
#   [B, 197, 768] -> [B, 12, 197, 64] @ [B, 12, 64, 197] @ [B, 12, 197, 64] -> [B, 197, 768]  多头
#----------------------------------------------------------#
class Attention(nn.Module):
    def __init__(
        self,
        dim,                # 输入token的dim=768
        num_heads=8,        # 分为8个head,类似组卷积
        qkv_bias=False,     # 是否使用q,k,v偏置
        qk_scale=None,      # 分母的 \sqrt d
        attn_drop_ratio=0., # attention dropout参数
        proj_drop_ratio=0., # 最后的dropout参数
    ):
        super().__init__()
        self.num_heads = num_heads
        # 每个head的dim维度,每个head均分dim
        head_dim = dim // num_heads
        #-----------------------------------------#
        #   分母的 \sqrt d
        #   如果传入了qk_scale就算,不然就计算: 根号下d_k
        #-----------------------------------------#
        self.scale = qk_scale or head_dim ** -0.5
        #-----------------------------------------#
        #   全连接层,直接使用一个全连接层实现求qkv,不使用3个,这样输出长度是3,和用3个一样
        #-----------------------------------------#
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        # 全连接层,处理拼接后的多头自注意力的输出
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [B, 197, 768] [B, position, channel]
        B, N, C = x.shape

        #-----------------------------------------------------------#
        #   qkv():   [B, 197, 768] -> [B, 197, 3 * 768]  最后维度变为3倍
        #   reshape: [B, 197, 3 * 768]   -> [B, 197, 3, 12, 64]   3指的是q,k,v; 12是heads个数, 64是每个head所占维度数量
        #   permute: [B, 197, 3, 12, 64] -> [3, B, 12, 197, 64]   3代表q,k,v
        #-----------------------------------------------------------#
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        #-----------------------------------------------------------#
        #   通过切片拿到q,k,v数据,切片之后第一个维度就没有了,绝了
        #   [B, 12, 197, 64]
        #-----------------------------------------------------------#
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        #-----------------------------------------------------------#
        #   针对每个heads的qk进行计算, qk都是4维数据,相乘是最后两个维度
        #   q乘以k的转置,(-2, -1)就是转置, 再除以根号下d_k
        #   [B, position, channel] @ [B, channel, position]
        #   从前面一个的position来看,一行代表一个position,它会乘以后面的每一列,后面的列也是position,就意味着求位置之间的相似度
        #
        #   [a, b] @ [B, a] = [a, a]  ab代表行列
        #   transpose: [B, 12, 197, 64] -> [B, 12, 64, 197]
        #   @:         [B, 12, 197, 64]  @ [B, 12, 64, 197] = [B, 12, 197, 197]
        #-----------------------------------------------------------#
        attn = (q @ k.transpose(-2, -1)) * self.scale
        #-----------------------------------------------------------#
        #   softmax处理, -1指的是在矩阵中跨列计算,将每一行的总和都设置为1; 取每一列,在行上做softmax
        #   ex: softmax([[1, 4],[5, 5.]]) = [[0.0474, 0.9526],[0.5000, 0.5000]]
        #   [B, 12, 197, 197] -> [B, 12, 197, 197]
        #-----------------------------------------------------------#
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # ❗❗❗important❗❗❗

        #-----------------------------------------------------------#
        #   针对每一个v的权重进行矩阵相乘
        #   @:          [B, 12, 197, 197] @ [B, 12, 197, 64] = [B, 12, 197, 64]  [197, 197]每行经过softmax和[192, 64]的每列相乘,后面的列代表位置,
        #   transpose:  [B, 12, 197, 64] -> [B, 197, 12, 64]
        #   reshape:    [B, 197, 12, 64] -> [B, 197, 768]
        #-----------------------------------------------------------#
        x = attn @ v
        x = x.transpose(1, 2)
        x = x.reshape(B, N, C)

        #-----------------------------------------------------------#
        #   全连接层,处理拼接后的多头自注意力的输出 [B, 197, 768] -> [B, 197, 768]
        #-----------------------------------------------------------#
        x = self.proj(x)
        x = self.proj_drop(x)   # ❗❗❗important❗❗❗
        return x


#-----------------------------------------------------------#
#   Encoder中的MLP
#   Linear -> GELU -> Dropout -> Linear -> Dropout  两个Dropout共用
#   第一个linear通道翻四倍,第二个Linear通道还原
#   [..., C] -> [..., n*C] -> [..., C]
#-----------------------------------------------------------#
class Mlp(nn.Module):
    """
    Transformer Encoder 中的Mlp Block
    """
    def __init__(
        self,
        in_features,             # in_features
        hidden_features=None,    # 一般是 in_features*4
        out_features=None,       # 一般等于 in_features
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        # 如果传入out_features,就使用,不然就是in_features
        out_features = out_features or in_features
        # 如果传入hidden_features,就使用,不然就是in_features
        hidden_features = hidden_features or in_features

        # 通道变为4倍
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # 通道还原
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):   # mix channel
        x = self.fc1(x)     # [B, 197, 768] -> [B, 197, 3072]
        x = self.act(x)     # ❗❗❗important❗❗❗
        x = self.drop(x)    # ❗❗❗important❗❗❗
        x = self.fc2(x)     # [B, 197, 3072] -> [B, 197, 768]
        x = self.drop(x)    # ❗❗❗important❗❗❗
        return x


#-----------------------------------------------------------#
#   ENcoder Block
#   Stage1: LN + Multi-Head + DropPath 有残差连接
#   Stage2: LN + MLP Block + DropPath 有残差连接
#   Transformer Encoder就是重复它
#-----------------------------------------------------------#
class Block(nn.Module):
    def __init__(
        self,
        dim,                    # 每个token的dim维度 768
        num_heads,              # multi-head中head个数 12
        mlp_ratio=4.,           # Mlp第一个linear的全连接层扩展倍数
        qkv_bias=False,         # 是否q,v,k的bias
        qk_scale=None,          # 是否使用qk_scale, \sqrt d
        drop_ratio=0.,          # Mlp最后的dropout比例
        attn_drop_ratio=0.,     # Attention中的attn的dropout比例
        drop_path_ratio=0.,     # 下面使用DropPath的比例
        act_layer=nn.GELU,      # 激活函数
        norm_layer=nn.LayerNorm # LN而不是BN
    ):
        super().__init__()

        #-----------------------------------------------------------#
        #   Stage1: LN + Multi-Head + DropPath 有残差连接
        #-----------------------------------------------------------#
        # Transformer Block第1个LayerNorm
        self.norm1 = norm_layer(dim)
        # Multi-Head
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio,
                              proj_drop_ratio=drop_ratio)

        # ratio大于0使用DropPath
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        #-----------------------------------------------------------#
        #   Stage2: LN + MLP Block + DropPath 有残差连接
        #-----------------------------------------------------------#
        # Transformer Block第2个LayerNorm
        self.norm2 = norm_layer(dim)

        # 计算Mlp中第一个Linear的out_feateures
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 没给out_features,使用 in_features
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))    # ❗❗❗important❗❗❗
        x = x + self.drop_path(self.mlp(self.norm2(x)))     # ❗❗❗important❗❗❗
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,              # kernel_size
            in_c=3,                     # in_channels
            num_classes=1000,           # num_classes
            embed_dim=768,              # Embedding维度
            depth=12,                   # Encoder Block重复次数
            num_heads=12,               # multi-head中head分类数
            mlp_ratio=4.0,              # Encoder Block第一个linear扩展倍数 4
            qkv_bias=True,              # 是否使用q,l,v偏置
            qk_scale=None,              # qkscale,不传入就自己计算
            representation_size=None,   # 最后nlp_head全连接层个数,如果不给,就不创建Pre-Logits
            distilled=False,            # 兼容搭建DeiT使用,这里用不到
            drop_ratio=0.,              # Mlp最后的dropout
            attn_drop_ratio=0.,         # Attention中的attn的dropout
            drop_path_ratio=0.,         # Block中使用DropPath
            embed_layer=PatchEmbed,     # 第一个Embedding
            norm_layer=None,
            act_layer=None
        ):

        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models 768
        #-----------------------------------------------------------#
        #   类别token,维度为1
        #-----------------------------------------------------------#
        self.num_tokens = 2 if distilled else 1
        #-----------------------------------------------------------#
        # 传入默认参数
        #-----------------------------------------------------------#
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # 激活函数默认使用ReLU
        act_layer = act_layer or nn.GELU
        #-----------------------------------------------------------#
        # Embedding [B, 3, 224, 224] -> [B, 196, 768]  [B, position, embedding]
        #-----------------------------------------------------------#
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        #-----------------------------------------------------------#
        # 获取pathes的个数,是14*14=196  一共196个方块
        #-----------------------------------------------------------#
        num_patches = self.patch_embed.num_patches
        #-----------------------------------------------------------#
        #   class token,需要和embedding后的结果拼接,
        #   形状是 [1, 1, 768] 第一个1是batch,不用管,方便拼接
        #-----------------------------------------------------------#
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 用不到,为None
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        #-----------------------------------------------------------#
        #   位置和 [1, 197, 768]相加
        #   和concat后数据形状一样,是 [1, 197, 768] 第一个1是batch,不用管
        #-----------------------------------------------------------#
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # 加上Position Embedding之后的dropout
        self.pos_drop = nn.Dropout(p=drop_ratio)

        #-----------------------------------------------------------#
        #   创建Encoder Block
        #   根据传入的drop_path_ratio,构建等差序列,长度就是深度,目的是为了DropPath的参数变化
        #-----------------------------------------------------------#
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)   # 重复depth次,这是一个列表生成式
        ])
        #-----------------------------------------------------------#
        #   通过Transformer Layer之后的LN
        #-----------------------------------------------------------#
        self.norm = norm_layer(embed_dim)

        #-----------------------------------------------------------#
        #   ImageNet21K上使用这个了,其余没使用
        #   Representation layer
        #-----------------------------------------------------------#
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            # 构建Pre Logits
            #   Linear + tanh
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            # 为空
            self.has_logits = False
            self.pre_logits = nn.Identity()

        #-----------------------------------------------------------#
        #   最后的Linear [B, 768] -> [B, num_classes]
        #-----------------------------------------------------------#
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # distilled为None,所以用不到
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, 3, 224, 224] -> [B, 196, 768] [B, position, channel]
        x = self.patch_embed(x)
        # [1, 1, 768] -> [B, 1, 768] 将维度为1的扩展
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        #-----------------------------------------------------------#
        # 维度分类参数,class拼在第0维
        # [B, 1, 768] cat [B, 196, 768] -> [B, 197, 768]
        #-----------------------------------------------------------#
        # Vit
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        # DeiT
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        #-----------------------------------------------------------#
        #   添加位置参数和dropout, 和位置参数相加
        #   [B, 197, 768] + [1, 197, 768] = [B, 197, 768]
        #-----------------------------------------------------------#
        x = self.pos_drop(x + self.pos_embed)   # ❗❗❗important❗❗❗

        #-----------------------------------------------------------#
        #   TransfromerBlock
        #   [B, 197, 768] -> [B, 197, 768]
        #-----------------------------------------------------------#
        x = self.blocks(x)
        x = self.norm(x)                        # ❗❗❗important❗❗❗

        #-----------------------------------------------------------#
        #   提取类别
        #-----------------------------------------------------------#
        if self.dist_token is None:
            return self.pre_logits(x[:, 0]) # 取出 [B, 197, 768] position中的 [B, 768]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        # [B, 3, 224, 224] -> [B, 768]
        x = self.forward_features(x)

        # Vit这里为None,不用看
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # [B, 768] -> [B, num_classes]
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    base large huge 主要区别是 embed_dim 768 1024 1280

    has_logits: ImageNet21K是Ture,其余是False
    patch16比patch32计算量要大,因为是卷积核大小,卷积核越小,计算量越大

    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    base large huge 主要区别是 embed_dim 768 1024 1280

    has_logits: ImageNet21K是Ture,其余是False
    patch16比patch32计算量要大,因为是卷积核大小,卷积核越小,计算量越大

    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    base large huge 主要区别是 embed_dim 768 1024 1280

    has_logits: ImageNet21K是Ture,其余是False
    patch16比patch32计算量要大,因为是卷积核大小,卷积核越小,计算量越大

    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    base large huge 主要区别是 embed_dim 768 1024 1280

    has_logits: ImageNet21K是Ture,其余是False
    patch16比patch32计算量要大,因为是卷积核大小,卷积核越小,计算量越大

    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    base large huge 主要区别是 embed_dim 768 1024 1280

    has_logits: ImageNet21K是Ture,其余是False
    patch16比patch32计算量要大,因为是卷积核大小,卷积核越小,计算量越大

    预训练权重是1个G
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = vit_base_patch16_224(num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]
