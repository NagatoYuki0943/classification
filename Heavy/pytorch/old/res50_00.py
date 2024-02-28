'''
ResNet50
[3, 4, 6, 3] = 16 * 3 + 2 = 50
'''

import  torch
from    torch import  nn
from    torch.nn import functional as F
from torchvision.models import resnet50

# 预训练模型
# model = resnet50(pretrained=True)


class Bottleneck(nn.Module):
    '''
    通道变化,宽高不变
    3层卷积
    '''

    def __init__(self, in_out_channels, middle_channels):
        '''

        :param in_out_channels: 输入和输出通道
        :param middle_channels: 中间计算通道
        '''
        super(Bottleneck, self).__init__()

        # Bottleneck(
        #       (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #     )

        # 三层卷积
        self.convs = nn.Sequential(
                    nn.Conv2d(in_out_channels, middle_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    #            通道为输入数据通道数
                    nn.BatchNorm2d(middle_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(middle_channels, middle_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(middle_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(middle_channels, in_out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    #            通道为输入数据通道数
                    nn.BatchNorm2d(in_out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
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
        out = F.relu(out, inplace=True)

        return out


class DownSampleBottleneck(nn.Module):
    '''
    通道变化,宽高不变或减半,根据stride判断
    3层卷积
    '''

    def __init__(self, in_channels, middle_channels, out_channels, stride: tuple):
        '''
        输入和输出通道可以不同,如果不同就让原始数据做一次卷积就行了
        图片的宽高不变,通过卷积核和padding调整
        :param in_channels:     输入通道
        :param middle_channels: 中间计算通道
        :param out_channels:    输出通道
        :param stride:  步长 (1, 1) 或 (2, 2) (1, 1)宽高不变,(2, 2) 宽高减半
        '''

        super(DownSampleBottleneck, self).__init__()

        # Bottleneck(
        #       (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        #       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #       (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       (relu): ReLU(inplace=True)
        #       (downsample): Sequential(
        #         (0): Conv2d(64, 256, kernel_size=(1, 1), stride=stride, bias=False)
        #         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #       )
        #     )


        # 三层卷积
        self.convs = nn.Sequential(
                    nn.Conv2d(in_channels, middle_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    #            通道为输入数据通道数
                    nn.BatchNorm2d(middle_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(middle_channels, middle_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
                    nn.BatchNorm2d(middle_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(middle_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
                    #            通道为输入数据通道数
                    nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                )


        # (downsample): Sequential(
        #   (0): Conv2d(64, 256, kernel_size=(1, 1), stride=stride, bias=False)
        #   (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )
        self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
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
        out = F.relu(out, inplace=True)

        return out



class ResNet50(nn.Module):
    def __init__(self, channels):
        super(ResNet50, self).__init__()

        # 第一层卷积
        # [b, 3, h, w] => [b, 64, h/4, w/4]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=1, ceil_mode=False)
        )


        # [b, 64, h, w] => [b, 256, h, w]
        # 3层
        self.layer1 = nn.Sequential(
            DownSampleBottleneck(64, 64, 256, (1, 1)),
        )
        self.__addBasicBlock(self.layer1, numbers=2, in_out_channels=256, middle_channels=64)


        # [b, 256, h, w] => [b, 512, h/2, w/2]
        # 4层
        self.layer2 = nn.Sequential(
            DownSampleBottleneck(256, 128, 512, (2, 2)),
        )
        self.__addBasicBlock(self.layer2, numbers=3, in_out_channels=512, middle_channels=128)


        # [b, 512, h, w] => [b, 1024, h/2, w/2]
        # 6层
        self.layer3 = nn.Sequential(
            DownSampleBottleneck(512, 256, 1024, (2, 2)),
        )
        self.__addBasicBlock(self.layer3, numbers=5, in_out_channels=1024, middle_channels=256)


        # [b, 1024, h, w] => [b, 2048, h/2, w/2]
        # 3层
        self.layer4 = nn.Sequential(
            DownSampleBottleneck(1024, 512, 2048, (2, 2)),
        )
        self.__addBasicBlock(self.layer4, numbers=2, in_out_channels=2048, middle_channels=512)


        # 将最后的宽高设置1
        # [b, 2048, h, w] => [b, 2048, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # [b, 2048] => [b, 10]
        self.outlayer = nn.Linear(2048, channels)    # 自带的bias为False,默认为True

        # 测试Bottleneck输出维度
        # tmp = torch.ones(2, 64, 32, 32)
        # out = self.Bottlenecks(tmp)
        # print(out.shape)
        # torch.Size([2, 512, 1, 1])



    def forward(self, x):
        '''
        :param x: [b, 3, h, w]
        :return:
        '''
        batch_size = x.size(0)

        # [b, 3, h, w] => [b, 64, h/4, w/4]
        x = self.conv1(x)

        # [b, 64,h/4, w/4] => [b, 2048, 1, 1]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)


        # flatten
        #  [b, 2048, 1, 1] =>  [b, 512]
        x = x.view(batch_size, -1)

        # [b, 2048] => [b, 10]
        x = self.outlayer(x)

        return x


    def __addBasicBlock(self, layer, numbers, in_out_channels, middle_channels):
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
            layer.add_module(f'Bottleneck{i+1}', Bottleneck(in_out_channels, middle_channels))


if __name__ == '__main__':


    # 测试Bottleneck
    # 维度不变
    x = torch.ones(2, 3, 32, 32)
    res_block = Bottleneck(3, 10)
    out = res_block(x)
    print(out.shape)
    # torch.Size([2, 3, 32, 32])

    print('*' * 50)

    # DownSampleBottleneck
    # 维度不变,宽高减半
    x = torch.ones(2, 3, 32, 32)
    down_sample_res_block = DownSampleBottleneck(3, 20, 10, (2, 2))
    out = down_sample_res_block(x)
    print(out.shape)
    # torch.Size([2, 10, 16, 16])

    print('*' * 50)



    # 测试 ResNet
    x = torch.ones(2, 3, 32, 32)
    res_net = ResNet50(10)
    out = res_net(x)
    print(out.shape)
    # torch.Size([2, 10])

    print('*' * 50)
    #print(res_net)

'''
ResNet50:
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer2): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer3): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (3): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (4): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (5): Bottleneck(
      (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (layer4): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
    (2): Bottleneck(
      (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)

'''