import torch
import torch.nn as nn
from torch.nn.modules.container import T

from torchvision import models
models.vgg11_bn()

from torchsummary import summary


class Vgg11_bn(nn.Module):
    def __init__(self, classes) -> None:
        super().__init__()

        # 8
        self.features = nn.Sequential(
            self.__make_layer(3,   64,  True),
            self.__make_layer(64,  128, True),
            self.__make_layer(128, 256, False),
            self.__make_layer(256, 256, True),
            self.__make_layer(256, 512, False),
            self.__make_layer(512, 512, True),
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
    model = Vgg11_bn(5)
    
    x = torch.ones(10, 3, 224, 224)
    y_predict = model(x)
    print(y_predict.size()) # torch.Size([10, 5])
    
    #print(model)

# VGG 11
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)          1
#     (2): ReLU(inplace=True)
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)     
#  
#     (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)         1
#     (6): ReLU(inplace=True)
#     (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)    
#   
#     (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)         0
#     (10): ReLU(inplace=True)

#     (11): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        1
#     (13): ReLU(inplace=True)
#     (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

#     (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (16): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        0
#     (17): ReLU(inplace=True)

#     (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (19): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        1
#     (20): ReLU(inplace=True)
#     (21): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

#     (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (23): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        0
#     (24): ReLU(inplace=True)

#     (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (26): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        1
#     (27): ReLU(inplace=True)
#     (28): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
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