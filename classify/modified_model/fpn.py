import torch
from torch import nn
import timm


# -------------------------------------------------#
#   卷积块
#   Conv2d + BatchNorm2d + LeakyReLU
# -------------------------------------------------#
def conv2d(in_channel, out_channel, kernel_size, stride=1, groups=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    groups = in_channel if in_channel == out_channel == groups else 1
    return nn.Sequential(
        nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
            groups=groups,
        ),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.1),
    )


class FPNDown(nn.Module):
    def __init__(
        self, channels: list[int], steps: int = 5, groups: bool = False
    ) -> None:
        """下采样FPN

        Args:
            channels (list[int]):    每层输出通道数
            steps (int, optional):   最终重复卷积次数. Defaults to 5.
            groups (bool, optional): 是否使用分组卷积. Defaults to False.
        """
        super().__init__()

        # 混合通道
        self.subs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            # 调整通道数和宽高
            self.subs.append(
                nn.Sequential(conv2d(channels[i], channels[i + 1], 1), nn.MaxPool2d(2))
            )
            # 调整后的数据和下一层相加后混合通道
            self.convs.append(
                nn.Sequential(conv2d(channels[i + 1], channels[i + 1], 3))
            )

        # 对最终的数据进行计算
        final_conv = []
        for i in range(steps):
            final_conv.append(
                conv2d(
                    channels[-1],
                    channels[-1],
                    kernel_size=3 if i % 2 == 0 else 1,
                    groups=channels[-1] if groups else 1,
                )
            )
        self.final_conv = nn.Sequential(*final_conv)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # 混合通道
        x_ = x[0]
        for i, (sub, conv) in enumerate(zip(self.subs, self.convs)):
            x_ = conv(sub(x_) + x[i + 1])

        # 对最终的数据进行计算
        x = self.final_conv(x_)
        return x


class FPNUp(nn.Module):
    def __init__(
        self, channels: list[int], steps: int = 5, groups: bool = False
    ) -> None:
        """上采样FPN

        Args:
            channels (list[int]):    每层输出通道数
            steps (int, optional):   最终重复卷积次数. Defaults to 5.
            groups (bool, optional): 是否使用分组卷积. Defaults to False.
        """
        super().__init__()
        # 反转顺序
        channels.reverse()

        # 混合通道
        self.ups = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            # 调整通道数和宽高
            self.ups.append(
                nn.Sequential(
                    conv2d(channels[i], channels[i + 1], 1),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                )
            )
            # 调整后的数据和下一层相加后混合通道
            self.convs.append(
                nn.Sequential(conv2d(channels[i + 1], channels[i + 1], 3))
            )

        # 对最终的数据进行计算
        final_conv = []
        for i in range(steps):
            final_conv.append(
                conv2d(
                    channels[-1],
                    channels[-1],
                    kernel_size=3 if i % 2 == 0 else 1,
                    groups=channels[-1] if groups else 1,
                )
            )
        self.final_conv = nn.Sequential(*final_conv)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # 反转顺序
        x.reverse()

        # 混合通道
        x_ = x[0]
        for i, (up, conv) in enumerate(zip(self.ups, self.convs)):
            x_ = conv(up(x_) + x[i + 1])

        # 对最终的数据进行计算
        x = self.final_conv(x_)
        return x


class FPNModel(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        out_indices: list[int] = [2, 3, 4],
        num_classes: int = 1000,
        steps: int = 5,
        groups: bool = False,
        fpn_mode: str = "up",
    ) -> None:
        """FPN

        Args:
            model_name (str, optional):        使用的模型名. Defaults to "resnet18".
            pretrained (bool, optional):       预训练模型. Defaults to True.
            out_indices (list[int], optional): 返回层数. Defaults to [2, 3, 4].
            num_classes (int, optional):       分类数. Defaults to 1000.
            steps (int, optional):             fpn最后层数. Defaults to 5.
            groups (bool, optional):           是否使用分组卷积. Defaults to False.
            fpn_mode (str, optional):          fpn方向, down or up. Defaults to "up".
        """
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )  # 一般选择后3层
        channels = self.model.feature_info.channels()  # [128, 256, 512]
        reduction = self.model.feature_info.reduction()  # [8, 16, 32]
        if fpn_mode == "up":
            self.fpn = FPNUp(channels, steps, groups)
        else:
            self.fpn = FPNDown(channels, steps, groups)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.fpn(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = FPNModel("convnext_tiny", out_indices=[1, 2, 3], groups=True, fpn_mode="up")
    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())
