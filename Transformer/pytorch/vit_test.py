import math
import torch
from torch import nn, Tensor
from timm.layers import DropPath
from functools import partial


device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class PatchEmbed(nn.Module):
    def __init__(self,
        in_channels: int,
        embedding_size: int,
        patch_size: int,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.conv = nn.Conv2d(in_channels, embedding_size, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)                    # [B, 3, 224, 224] -> [B, 768, 14, 14]
        x = x.flatten(2).permute(0, 2, 1)   # [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
        x = self.norm(x)                    # ❗❗❗important❗❗❗
        return x


def test_patch_embed():
    B, C, H, W = 10, 3, 224, 224
    x = torch.ones(B, C, H, W).to(device)
    patch_embed = PatchEmbed(3, 768, 16).to(device)
    patch_embed.eval()
    with torch.inference_mode():
        y = patch_embed(x)
    print(y.size())


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop_ratio: float = 0.,
        proj_drop_ratio: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        qkv: Tensor = self.qkv(x)                               # [B, N, C] -> [B, N, 3*C]
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)  # [B, N, 3*C] -> [B, N, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()           # [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        q: Tensor
        k: Tensor
        v: Tensor
        q, k, v = qkv.unbind(0)                                 # [3, B, num_heads, N, head_dim] -> 3 * [B, num_heads, N, head_dim]

        attn: Tensor = q @ k.transpose(-1, -2) / self.scale # [B, num_heads, N, head_dim] @ [B, num_heads, head_dim, N] = [B, num_heads, N, N]
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn) # ❗❗❗important❗❗❗

        x = attn @ v                # [B, num_heads, N, N] @ [B, num_heads, N, head_dim] = [B, num_heads, N, head_dim]
        x = x.transpose(1, 2)       # [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim]
        x = x.reshape(B, N, C)      # [B, N, num_heads, head_dim] -> [B, N, C]

        x = self.proj(x)            # [B, N, C] -> [B, N, C]
        x = self.proj_drop(x)       # ❗❗❗important❗❗❗
        return x


def test_attn():
    B, N, C = 10, 196, 768
    x = torch.ones(B, N, C).to(device)
    attn = Attention(768, 12).to(device)
    attn.eval()
    with torch.inference_mode():
        y = attn(x)
    print(y.size())


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int = 4,
        act_layer = nn.GELU,
        norm_layer=None,
        bias=True,
        drop_ratio: float = 0.,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_ratio)
        self.norm = norm_layer(hidden_dim) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop2 = nn.Dropout(drop_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)     # ❗❗❗important❗❗❗
        x = self.drop1(x)   # ❗❗❗important❗❗❗
        x = self.norm(x)    # ❗❗❗important❗❗❗
        x = self.fc2(x)
        x = self.drop2(x)   # ❗❗❗important❗❗❗
        return x


def test_mlp():
    B, N, C = 10, 196, 768
    x = torch.ones(B, N, C).to(device)
    mlp = Mlp(768).to(device)
    mlp.eval()
    with torch.inference_mode():
        y = mlp(x)
    print(y.size())


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self.gamma * x  # [B, N, C] * [C] = [B, N, C]
        return x


class TransfromerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop_ratio: float = 0.,
        proj_drop_ratio: float = 0.,
        drop_path_ratio: float = 0.,
        mlp_ratio: int = 4,
        mlp_drop_ratio: float = 0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop_ratio, proj_drop_ratio)
        self.ls1 = LayerScale(dim)  # 原本vit没用layerscale,自己添加的
        self.drop_path1 = DropPath(drop_path_ratio)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_ratio, act_layer, drop_ratio=mlp_drop_ratio)
        self.ls2 = LayerScale(dim)
        self.drop_path2 = DropPath(drop_path_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x)))) # ❗❗❗important❗❗❗
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))  # ❗❗❗important❗❗❗
        return x


def test_transformer():
    B, N, C = 10, 196, 768
    x = torch.ones(B, N, C).to(device)
    transoformer = TransfromerBlock(768, 12).to(device)
    transoformer.eval()
    with torch.inference_mode():
        y = transoformer(x)
    print(y.size())


class VisionTransfromer(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        num_blocks: int = 12,
        num_heads: int = 12,
        qkv_bias: bool = True,
        position_drop_ratio: float = 0.,
        attn_drop_ratio: float = 0.,
        proj_drop_ratio: float = 0.,
        drop_path_ratio: float = 0.,
        mlp_ratio: int = 4,
        mlp_drop_ratio: float = 0.,
        act_layer = nn.GELU,
        norm_layer = None,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.use_cls_token = use_cls_token

        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # PatchEmbed
        self.patch_embed = PatchEmbed(in_channels, dim, patch_size)

        # cls_token & position_embed
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.position_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        else:
            self.position_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.position_drop = nn.Dropout(position_drop_ratio)

        # TransfromerBlock
        # drop_path_ratio 等差数列
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, num_blocks)]
        blocks = [TransfromerBlock(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=proj_drop_ratio,
            drop_path_ratio=dpr[i],
            mlp_ratio=mlp_ratio,
            mlp_drop_ratio=mlp_drop_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )   for i in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(dim)

        # linear
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        B, *_ = x.shape

        # PatchEmbed
        x = self.patch_embed(x) # [B, 3, 224, 224] -> [B, 196, 768]

        # cls_token & position_embed
        if self.use_cls_token:
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1) # [B, 1, 768] cat [B, 196, 768] -> [B, 197, 768]
        x = x + self.position_embed # [B, 197, 768] + [B, 197, 768] = [B, 197, 768]
        x = self.position_drop(x)   # ❗❗❗important❗❗❗

        # TransfromerBlock
        x = self.blocks(x)      # [B, 197, 768] -> [B, 197, 768]
        x = self.norm(x)        # ❗❗❗important❗❗❗

        # get classes layer
        if self.use_cls_token:
            x = x[:, 0, :]      # [B, 197, 768] get [B, 768]
        else:
            x = x.mean(dim=1)   # [B, 196, 768] -> [B, 768]

        # linear
        x = self.fc(x)          # [B, 768] -> [B, num_classes]
        return x


def vit_base_patch16_224(num_classes: int = 1000):
    model = VisionTransfromer(
        num_classes=num_classes,
        image_size=224,
        patch_size=16,
        dim=768,
        num_blocks=12,
        num_heads=12,
        use_cls_token=True
    )
    return model


def test_vit():
    B, C, H, W = 10, 3, 224, 224
    x = torch.ones(B, C, H, W).to(device)
    vit = vit_base_patch16_224(5).to(device)

    vit.eval()
    with torch.inference_mode():
        y = vit(x)
    print(y.size()) # [10, 5]


if __name__ == "__main__":
    # test_patch_embed()
    # test_attn()
    # test_mlp()
    # test_transformer()
    test_vit()
