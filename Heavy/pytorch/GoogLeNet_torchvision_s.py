"""
大小不用指定为 299 299

图像标准化(没有使用预训练模型,不这样也行)
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


辅助输出的形状和最终输出的形状完全相同,形状都是[batch, num_classes];只在训练时会有,验证时没有


# 训练时创建模型 aux_logits=True init_weights=True
net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True).to(device)

# 训练时使用3个返回值
model.train()
for step, images, labels in enumerate(train_data):
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    logits, aux_logits2, aux_logits1 = net(images)
    loss0 = loss_function(logits, labels.to(device))
    loss1 = loss_function(aux_logits1, labels.to(device))
    loss2 = loss_function(aux_logits2, labels.to(device))
    loss = loss0 + loss1 * 0.3 + loss2 * 0.3
    loss.backward()
    optimizer.step()


# 测试时创建模型 aux_logits=False
model = GoogLeNet(num_classes=5, aux_logits=False).to(device)
missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

# 测试时只有一个输出
model.eval()
logits = net(images)

"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.types import Number
from torchvision import models


class BasicConv2d(nn.Module):
    """
    Conv + ReLU 没有BN
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    """
    所有的Inception都是相同的
    所有branch的宽高都不变,是输入的宽高,维度会变化
    """

    def __init__(
        self,
        in_channels,  # in_channel
        ch1x1,  # 1x1 out_channel
        ch3x3red,  # 3x3 middle_channel
        ch3x3,  # 3x3 out_channel
        ch5x5red,  # 5x5 middle_channel
        ch5x5,  # 5x5 out_channel
        pool_proj,
    ):  # max_pool out_channel
        super().__init__()

        # 1x1
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 3x3
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(
                ch3x3red, ch3x3, kernel_size=3, padding=1
            ),  # 保证输出大小等于输入大小
        )

        # 5x5
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(
                ch5x5red, ch5x5, kernel_size=5, padding=2
            ),  # 保证输出大小等于输入大小
        )

        # max_pool 宽高不变,因为k=3 s=1 p=1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """
    辅助分类层
    辅助输出的形状和最终输出的形状完全相同,形状都是[batch, num_classes];只在训练时会有,验证时没有
    """

    def __init__(
        self,
        in_channels,  # 输入通道个数
        num_classes,
    ):  # 最后分类数
        super().__init__()
        self.averagePool = nn.AvgPool2d(
            kernel_size=5, stride=3
        )  # 池化  [N, C, 14, 14] => [N, C, 4, 4]
        self.conv = BasicConv2d(
            in_channels, 128, kernel_size=1
        )  # 卷积  output[N, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)  # 128*4*4=2048
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1(inception4a): [N, 512, 14, 14] => [N, 512,  4,  4]
        # aux2(inception4d): [N, 528, 14, 14] => [N, 528,  4,  4]
        x = self.averagePool(x)

        # aux1(inception4a): [N, 512, 4, 4] => [N, 128,  4,  4]
        # aux2(inception4d): [N, 528, 4, 4] => [N, 128,  4,  4]
        x = self.conv(x)

        # [N, 128, 4, 4] => [N, 128*4*4]
        x = torch.flatten(x, 1)

        # 50%随机效果失活      training会变化 train模式下为True,eval模式下为False
        x = F.dropout(x, 0.5, training=self.training)

        # [N, 2048] => [N, 1024]
        x = F.relu(self.fc1(x), inplace=True)
        # 50%随机效果失活
        x = F.dropout(x, 0.5, training=self.training)
        # [N, 1024] => [N, num_classes]
        x = self.fc2(x)

        return x


class GoogLeNet(nn.Module):
    def __init__(
        self,
        num_classes=1000,  # 分类个数
        aux_logits=True,  # 是否使用辅助分类器
        init_weights=False,
    ):  # 是否初始化权重
        super().__init__()
        self.aux_logits = aux_logits

        # [N, 3, 224, 224] => [N, 64, 112, 112]
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # [N, 64, 112, 112] => [N, 64, 56, 56]
        self.maxpool1 = nn.MaxPool2d(
            3, stride=2, ceil_mode=True
        )  # 参数ceil_mode: 如果下采样最右边和下边的不足一组,不会丢弃,而是继续使用
        # [N, 64, 56, 56] => [N, 64, 56, 56]
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        # [N, 64, 56, 56] => [N, 192, 56, 56]
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        # [N, 192, 56, 56] => [N, 192, 28, 28]
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # in_channel 1x1out_channel 3x3middle_channel 3x3out_channel 5x5middle_channel 5x5out_channel max_poolout_channel
        # [N, 192, 28, 28] => [N, 256, 28, 28]
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # [N, 256, 28, 28] => [N, 480, 28, 28]
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # [N, 480, 28, 28] => [N, 480, 14, 14]
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # [N, 480, 14, 14] => [N, 512, 14, 14]
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        # [N, 512, 14, 14] => [N, 512, 14, 14]
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # [N, 512, 14, 14] => [N, 512, 14, 14]
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # [N, 512, 14, 14] => [N, 528, 14, 14]
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # [N, 528, 14, 14] => [N, 832, 14, 14]
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # [N, 832, 14, 14] => [N, 832, 7, 7]
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # [N, 832, 7, 7] => [N, 832, 7, 7]
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # [N, 832, 7, 7] => [N, 1024, 7, 7]
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 两个辅助分类器 inception4a inception4d
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        # [N, 1024, 7, 7] => [N, 1024, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        # [N, 1024] => [N, num_classes]
        self.fc = nn.Linear(1024, num_classes)

        # 初始化权重
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # [N, 3, 224, 224] => [N, 64, 112, 112]
        x = self.conv1(x)
        # [N, 64, 112, 112] => [N, 64, 56, 56]
        x = self.maxpool1(x)
        # [N, 64, 56, 56] => [N, 64, 56, 56]
        x = self.conv2(x)
        # [N, 64, 56, 56] => [N, 192, 56, 56]
        x = self.conv3(x)
        # [N, 192, 56, 56] => [N, 192, 28, 28]
        x = self.maxpool2(x)

        # [N, 192, 28, 28] => [N, 256, 28, 28]
        x = self.inception3a(x)
        # [N, 256, 28, 28] => [N, 480, 28, 28]
        x = self.inception3b(x)
        # [N, 480, 28, 28] => [N, 480, 14, 14]
        x = self.maxpool3(x)

        # [N, 480, 14, 14] => [N, 512, 14, 14]
        x = self.inception4a(x)
        # [N, 512, 14, 14]
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux1(x)

        # [N, 512, 14, 14] => [N, 512, 14, 14]
        x = self.inception4b(x)
        # [N, 512, 14, 14] => [N, 512, 14, 14]
        x = self.inception4c(x)
        # [N, 512, 14, 14] => [N, 528, 14, 14]
        x = self.inception4d(x)
        # [N, 528, 14, 14]
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux2(x)

        # [N, 528, 14, 14] => [N, 832, 14, 14]
        x = self.inception4e(x)
        # [N, 832, 14, 14] => [N, 832, 7, 7]
        x = self.maxpool4(x)
        # [N, 832, 7, 7] => [N, 832, 7, 7]
        x = self.inception5a(x)
        # [N, 832, 7, 7] => [N, 1024, 7, 7]
        x = self.inception5b(x)

        # [N, 1024, 7, 7] => [N, 1024, 1, 1]
        x = self.avgpool(x)
        # [N, 1024, 1, 1] => [N, 1024]
        x = torch.flatten(x, 1)
        # [N, 1024] => [N, num_classes]
        x = self.dropout(x)
        x = self.fc(x)
        # [N, num_classes]

        # 训练模式同时aux_logits=True才返回辅助分类器
        #  model.train() model.eval()
        if self.training and self.aux_logits:  # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    x = torch.ones(1, 3, 224, 224).to(device)
    model = GoogLeNet(num_classes=10, init_weights=True).to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size())
    # torch.Size([2, 10])
