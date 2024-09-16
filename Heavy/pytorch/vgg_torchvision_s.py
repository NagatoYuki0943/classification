"""
vgg11 13 16 19  16最常见
数据量太大,建议使用迁移学习

预训练模型和官网的有些不同
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


# official pretrain weights
model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # Dropout减少过拟合
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # Dropout减少过拟合
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积初始化
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 偏置设置为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 线性层初始化
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                # 偏置设置为0
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        # 取出out_channels

        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            # 交换in_channels
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    # 数字代表卷积核个数(就是out_channels, M代表池化层
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def vgg(model_name="vgg16", pretrained=False, **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(
        model_name
    )
    cfg = cfgs[model_name]
    model = VGG(make_features(cfg), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = vgg(model_name="vgg16", pretrained=False)
    # 修改最后的全连接层
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 5)
    model.to(device)

    # 冻结权重
    for k, v in model.named_parameters():
        if "features" in k:
            # v.requires_grad = False
            v.requires_grad_(False)
        else:
            print(k)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # [1, 5]
