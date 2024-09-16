"""
噼里啪啦Wz修改过,准确率没有降低,训练加快
"""

import torch
import torch.nn as nn
from torchvision.models import alexnet


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 核心数降低 96 => 48
            nn.Conv2d(
                3, 48, kernel_size=11, stride=4, padding=2
            ),  # input[3, 224, 224]  output[48, 55, 55]  右下不够就舍弃
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        初始化权重
        """
        # 遍历所有的层结构
        for m in self.modules():
            # 是属于哪一个类别
            if isinstance(m, nn.Conv2d):
                # 初始化数据 normal 正态分布
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # 偏置初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # normal 正态分布
                nn.init.normal_(m.weight, 0, 0.01)
                # 偏置设置为0
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = AlexNet(num_classes=5).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())  # torch.Size([1, 5])

    print(model)
    # AlexNet(
    #   (features): Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    #     (1): ReLU(inplace=True)
    #     (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

    #     (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    #     (4): ReLU(inplace=True)
    #     (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

    #     (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (7): ReLU(inplace=True)

    #     (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (9): ReLU(inplace=True)

    #     (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (11): ReLU(inplace=True)
    #     (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #   )
    #   (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
    #   (classifier): Sequential(
    #     (0): Dropout(p=0.5, inplace=False)
    #     (1): Linear(in_features=9216, out_features=4096, bias=True)
    #     (2): ReLU(inplace=True)

    #     (3): Dropout(p=0.5, inplace=False)
    #     (4): Linear(in_features=4096, out_features=4096, bias=True)
    #     (5): ReLU(inplace=True)

    #     (6): Linear(in_features=4096, out_features=1000, bias=True)
    #   )
    # )
