import torch
import torch.nn as nn
from torch.nn.modules.container import T

from torchvision import models
models.vgg16_bn()

from torchsummary import summary


class Vgg16_bn(nn.Module):
    def __init__(self, classes) -> None:
        super().__init__()

        # 13
        self.features = nn.Sequential(
            self.__make_layer(3,   64,  False),
            self.__make_layer(64,  64,  True),

            self.__make_layer(64, 128,  False),
            self.__make_layer(128, 128, True),

            self.__make_layer(128, 256, False),
            self.__make_layer(256, 256, False),
            self.__make_layer(256, 256, True),

            self.__make_layer(256, 512, False),
            self.__make_layer(512, 512, False),
            self.__make_layer(512, 512, True),
            self.__make_layer(512, 512, False),
            self.__make_layer(512, 512, False),
            self.__make_layer(512, 512, True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # 3
        self.classifier = nn.Sequential(
            # 7 * 7 * 512 = 25088
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, classes, bias=True)
        )


    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        output = self.classifier(x)
        return output



    def __make_layer(self,in_channels, out_channels, max_pool=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        layers.append(nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # 添加池化层
        if max_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))

        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = Vgg16_bn(5)
    
    x = torch.ones(10, 3, 224, 224)
    y_predict = model(x)
    print(y_predict.size()) # torch.Size([10, 5])
    print('*' * 50)
    print(model)

# VGG16_BN
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)      0
#     (2): ReLU(inplace=True)

#     (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)      1
#     (5): ReLU(inplace=True)
#     (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)       


#     (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)     0   
#     (9): ReLU(inplace=True)

#     (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    1
#     (12): ReLU(inplace=True)
#     (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)


#     (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    0
#     (16): ReLU(inplace=True)

#     (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    0
#     (19): ReLU(inplace=True)

#     (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    1
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)


#     (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    0
#     (26): ReLU(inplace=True)

#     (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    0
#     (29): ReLU(inplace=True)

#     (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    1
#     (32): ReLU(inplace=True)
#     (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

#     (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    0
#     (36): ReLU(inplace=True)

#     (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    0
#     (39): ReLU(inplace=True)

#     (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    1
#     (42): ReLU(inplace=True)
#     (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)

#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)

#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )