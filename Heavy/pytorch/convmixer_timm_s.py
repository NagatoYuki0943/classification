""" ConvMixer

"""
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d
from timm.models._registry import register_model, generate_default_cfgs
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq

__all__ = ['ConvMixer']


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


#---------------------------#
# 每个block块的结构
#            │
#   ┌──────-─┤
#   │        │
#   │     conv(7x7) / (9x9)   使用了DW卷积
#   │        |
#   │       act
#   │        │
#   │        bn
#   │        │
#   └───────add
#            │
#         conv(1x1)
#            │
#           act
#            │
#            bn
#            │
#---------------------------#
class ConvMixer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            kernel_size=9,
            patch_size=7,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            drop_rate=0.,
            act_layer=nn.GELU,
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dim
        self.grad_checkpointing = False

        # [B, 3, 224, 224] -> [B, 768, 32, 32]
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            act_layer(),
            nn.BatchNorm2d(dim)
        )

        # [B, 768, 32, 32] -> [B, 768, 32, 32]
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        act_layer(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    act_layer(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.pooling = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)        # [B, 3, 224, 224] -> [B, 768, 32, 32]
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)  # [B, 768, 32, 32] -> [B, 768, 32, 32]
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.pooling(x)                     # [B, 768, 32, 32] -> [B, 768, 1, 1] -> [B, 768]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)# [B, 768] -> [B, num_classes]

    def forward(self, x):
        x = self.forward_features(x)    # [B, 3, 224, 224] -> [B, 768, 32, 32]
        x = self.forward_head(x)        # [B, 768, 32, 32] -> [B, num_classes]
        return x


def _create_convmixer(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ConvMixer, variant, pretrained, **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        'first_conv': 'stem.0',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'convmixer_1536_20.in1k': _cfg(hf_hub_id='timm/'),
    'convmixer_768_32.in1k': _cfg(hf_hub_id='timm/'),
    'convmixer_1024_20_ks9_p14.in1k': _cfg(hf_hub_id='timm/')
})


def convmixer_1536_20(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=1536, depth=20, kernel_size=9, patch_size=7, **kwargs)
    return _create_convmixer('convmixer_1536_20', pretrained, **model_args)


def convmixer_768_32(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, act_layer=nn.ReLU, **kwargs)
    return _create_convmixer('convmixer_768_32', pretrained, **model_args)


def convmixer_1024_20_ks9_p14(pretrained=False, **kwargs) -> ConvMixer:
    model_args = dict(dim=1024, depth=20, kernel_size=9, patch_size=14, **kwargs)
    return _create_convmixer('convmixer_1024_20_ks9_p14', pretrained, **model_args)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = convmixer_768_32(pretrained=False, num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 5]

    if False:
        onnx_path = 'convmixer_768_32.onnx'
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
