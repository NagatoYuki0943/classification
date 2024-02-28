'''
resnet18
[2, 2, 2, 2] = 8 * 2 + 2 = 16


初始 7*7 替换为 3个 3*3 
下采样由 Conv 转换为 AvgPool2d Conv会损失0.75数据
DownSampleBasicBlock卷积改变
ReLU替换为LeakyReLU
BatchNorm2d,LeakyReLU提到Conv前面来
'''

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


# 预训练模型
# model = resnet18(pretrained=True)


class BasicBlock(nn.Module):
    '''
    通道变化,宽高不变
    2层卷积
    '''

    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()

        # 两层卷积
        self.convs = nn.Sequential(
            #            通道为输入数据通道数
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),


            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )


    def forward(self, x):
        """
        :param x: [b, c, h, w]
        :return:
        """

        # 卷积
        out = self.convs(x)

        # shortcut短接
        out = x + out

        # 要进行relu
        F.leaky_relu(out, inplace=True)

        return out


class DownSampleBasicBlock(nn.Module):
    '''
    通道变化,宽高减半
    2层卷积
    '''

    def __init__(self, in_channels, out_channels):
        '''
        输入和输出通道可以不同,如果不同就让原始数据做一次卷积就行了
        图片的宽高不变,通过卷积核和padding调整
        :param in_channels:
        :param out_channels:
        '''
        super(DownSampleBasicBlock, self).__init__()

        # DownSampleBasicBlock(
        #       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        #       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (downsample): Sequential(
        #         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       )
        #     )

        # 两层卷积
        self.convs = nn.Sequential(


            # 增加 1*1 卷积
            #            通道为输入数据通道数
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            

            #            通道为输入数据通道数
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            

            # 3*3 => 1*1
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False), 
        )


        #   # 有下采样
        #   (downsample): Sequential(
        #     (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #     (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        self.downsample = nn.Sequential(
            # 增加下采样,不使用卷积将宽高减半
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),

            #            通道为输入数据通道数
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )


    def forward(self, x):
        """
        :param x: [b, c, h, w]
        :return:
        """

        # 卷积
        out = self.convs(x)

        # 宽高减半
        x = self.downsample(x)

        # shortcut短接
        out = x + out

        # 要进行relu
        out = F.leaky_relu(out, inplace=True)

        return out


class ResNet18(nn.Module):
    def __init__(self, channels):
        super(ResNet18, self).__init__()

        # 第一层卷积
        # [b, 3, h, w] => [b, 64, h/4, w/4]
        self.conv1 = nn.Sequential(
            # 7*7 变为 3 层 3*3
            nn.Conv2d( 3, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        # [b, 64, h, w] => [b, 64, h, w]
        self.layer1 = nn.Sequential(
            BasicBlock(64),
            BasicBlock(64),
        )

        # [b, 64, h, w] => [b, 128, h/2, w/2]
        self.layer2 = nn.Sequential(
            DownSampleBasicBlock(64, 128),
            BasicBlock(128),
        )

        # [b, 128, h, w] => [b, 256, h/2, w/2]
        self.layer3 = nn.Sequential(
            DownSampleBasicBlock(128, 256),
            BasicBlock(256),
        )

        # [b, 256, h, w] => [b, 512, h/2, w/2]
        self.layer4 = nn.Sequential(
            DownSampleBasicBlock(256, 512),
            BasicBlock(512),
        )

        # 将最后的宽高设置1
        # [b, 512, h, w] => [b, 512, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # [b, 512] => [b, 10]
        self.outlayer = nn.Linear(512, channels)  # 自带的bias为False,默认为True

        # 测试BasicBlock输出维度
        # tmp = torch.ones(2, 64, 32, 32)
        # out = self.BasicBlocks(tmp)
        # print(out.shape)
        # torch.Size([2, 512, 1, 1])


    def forward(self, x):
        '''
        :param x: [b, 3, h, w]
        :return:
        '''
        batch_size = x.size(0)

        # [b, 3, h, w] => [b, 64, h, w]
        x = self.conv1(x)

        # [b, 64, h, w] => [b, 512, 1, 1]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # flatten
        #  [b, 512, 1, 1] =>  [b, 512]
        x = x.view(batch_size, -1)

        # [b, 512] => [b, 10]
        x = self.outlayer(x)

        return x

    def _addBasicBlock(self, layer, numbers, in_out_channels):
        '''
        向layer中添加Bottleneck层
        :param layer:  要添加Bottleneck的layer
        :param numbers: 添加层数
        :param in_out_channels: Bottleneck的参数
        :param middle_channels: Bottleneck的参数
        :return:
        '''
        for i in range(numbers):
            #                参数1名字必须不同,不然会被覆盖
            layer.add_module(f'BasicBlock{i + 1}', BasicBlock(in_out_channels))


if __name__ == '__main__':
    # 测试BasicBlock
    # 维度不变
    x = torch.ones(2, 3, 32, 32)
    res_block = BasicBlock(3)
    out = res_block(x)
    print(out.shape)
    # torch.Size([2, 3, 32, 32])

    print('*' * 50)

    # DownSampleBasicBlock
    # 维度不变,宽高减半
    x = torch.ones(2, 3, 32, 32)
    down_sample_res_block = DownSampleBasicBlock(3, 10)
    out = down_sample_res_block(x)
    print(out.shape)
    # torch.Size([2, 10, 16, 16])

    print('*' * 50)

    # 测试 ResNet
    x = torch.ones(2, 3, 32, 32)
    res_net = ResNet18(10)
    out = res_net(x)
    print(out.shape)
    # torch.Size([2, 10])

    print('*' * 50)
    # print(res_net)

