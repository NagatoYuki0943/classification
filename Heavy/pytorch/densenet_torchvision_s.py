'''
densenet自带的预训练模型没法使用torch.load()加载,不过自己训练的模型可以加载
121 169 201 161
以上四个结构主要差别是denseblock里面重复的_DenseBlock()次数不同

DenseLayer中拼接之前所有输出的方法使用的系统方法

DenseNet(
    (features): Sequential(
        (conv0)
        (norm0)
        (relu0)
        (pool0)
        (denseblock1): DenseLayer1 * n    维度变化,高宽不变
        (transition1): DenseLayer1 * 1    维度,高宽减半
        (denseblock2): DenseLayer1 * n    维度变化,高宽不变
        (transition2): DenseLayer1 * 1    维度,高宽减半
        (denseblock3): DenseLayer1 * n    维度变化,高宽不变
        (transition3): DenseLayer1 * 1    维度,高宽减半
        (denseblock4): DenseLayer1 * n    维度变化,高宽不变
        (norm5): BatchNorm2d()
    )
    (classifier): Linear()
'''


from typing import Any, List, Tuple
from collections import OrderedDict
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torch.hub import load_state_dict_from_url


class _DenseLayer(nn.Module):
    '''
    单个的稠密卷积层
    宽高不变,输出维度增长倍率是32,所以 out_channel = 32
    BN + ReLU + 1x1Conv + BN + ReLU + 3x3Conv
    '''
    def __init__(self,
                 input_c: int,      # 输入维度
                 growth_rate: int,  # 32 增长倍率,中间维度是增长倍率的bn_size倍,输出维度也是增长倍率
                 bn_size: int,      # 4
                 drop_rate: float,  # 最后舍弃一些特征的比例
                 memory_efficient: bool = False): # 节省内存
        super().__init__()

        # 降低输出的维度,因为拼接多次都层数太多,转换为bn_size(4) * growth_rate(32), 转换为128层
        # [b, n, h, w] => [b, 128, h, w]
        self.add_module("norm1", nn.BatchNorm2d(input_c))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels=input_c,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))

        # [b, 128, h, w] => [b, 32, h, w]
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False))
        # 最后舍弃一些特征
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        # 传送过来的是数组,要先在第二维拼接,然后降低维度到128
        concat_features = torch.cat(inputs, 1)
        # [b, n, h, w] => [b, 128, h, w]
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        '''
        查看数据是否需要求导
        '''
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    # 使用装饰器调用bn_function
    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        # 传送过来的是数组,要现在第二维拼接,然后降低维度到128
        # [b, n, h, w] => [b, 128, h, w]
        if self.memory_efficient and self.any_requires_grad(prev_features):
            '''
            节省内存并向前传播
            '''
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        # [b, 128, h, w] => [b, 32, h, w]
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        # 舍弃一些特征
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    '''
    创建DenseBlock,每个DenseBlock都有n个DenseLayer
    将Input放进列表features中,将它作为参数传入DenseLayer,然后将output添加到features列表中,循环放入DenseLayer,最后获得输入+全部输出列表,然后拼接到一起并返回
    最后的输出维度是 out_channel =  input_c + num_layers * growth_rate
    '''
    _version = 2

    def __init__(self,
                 num_layers: int,       # DenseLayer重复次数
                 input_c: int,          # 输出维度
                 bn_size: int,          # 4 growth_rate * bn_size就是DenseLayer中降低的维度
                 growth_rate: int,      # 32 增长率,每一个DenseLayer的输出维度
                 drop_rate: float,
                 memory_efficient: bool = False):
        super().__init__()
        # 构建多个DenseLayer
        for i in range(num_layers):
            layer = _DenseLayer(input_c + i * growth_rate,  # 调整输入的维度,初始维度加上之前所有DenseLayer的输入维度
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        # 将输入放进要放回的列表中
        features = [init_features]

        for name, layer in self.items():
            # 循环使用DenseLayer,参数features是一个数组,所以在DenseLayer中要先拼接数据
            new_features = layer(features)
            # 将输出保存到数组中
            features.append(new_features)

        # 将开始输入和多个层的输出拼接到一起
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    '''
    两层DenseBlock之间的降采样
    通道和宽高都减半
    BN + ReLU + Conv(out = in / 2) + Pool(k=s=2)
    '''
    def __init__(self,
                 input_c: int,
                 output_c: int):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(input_c))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(input_c,                      # 通道减半 out = in / 2
                                          output_c,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))  # 宽高减半


class DenseNet(nn.Module):
    """
    Densenet-BC model class for imagenet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient
    """
    def __init__(self,
                 growth_rate: int = 32,         # 32 增长率,每一个DenseLayer的输出维度
                 block_config: Tuple[int, int, int, int] = (6, 12, 24, 16), # 四个DenseBlock中DenseLayer的重复次数
                 num_init_features: int = 64,   # in_channels
                 bn_size: int = 4,              # 4 growth_rate * bn_size就是DenseLayer中降低的维度
                 drop_rate: float = 0,          # DenseLayer中最后的Dropout的丢弃参数
                 num_classes: int = 1000,       # 分类数
                 memory_efficient: bool = False):   # 节省内存
        super().__init__()

        # 最开始特征处理 first conv+bn+relu+pool
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),    # kernel_size和stride可以不同 这时stride的一步就是一个kernel_size大小,通过padding可以让效果和kernel=stride相同
        ]))

        num_features = num_init_features    # num_features: 最开始特征数

        # 共4个DenseBlock
        for i, num_layers in enumerate(block_config):
        # num_layers是DenseLayer重复次数
            block = _DenseBlock(num_layers=num_layers,      # 重复DenseLayer次数
                                input_c=num_features,       # in_channels
                                bn_size=bn_size,            # 4 growth_rate * bn_size就是DenseLayer中降低的维度
                                growth_rate=growth_rate,    # 32 增长率,每一个DenseLayer的输出维度
                                drop_rate=drop_rate,        # DenseLayer中最后的Dropout的丢弃参数
                                memory_efficient=memory_efficient)
            # 添加到字典中
            self.features.add_module("denseblock%d" % (i + 1), block)

            # DenseBlock最后的输出维度是 input_c + num_layers * growth_rate
            num_features = num_features + num_layers * growth_rate

            # 不是最后一个block,就在block之间添加Transition, 降低维度和高宽
            if i != len(block_config) - 1:
                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2)
                # 添加到字典中
                self.features.add_module("transition%d" % (i + 1), trans)
                # 修改最终的num_features
                num_features = num_features // 2

        # 最后的Norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # 线性层
        self.classifier = nn.Linear(num_features, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))    # 7x7 => 1x1
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
}


def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet("densenet121", 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet("densenet169", 32, (6, 12, 32, 32), 64, pretrained, progress, **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet("densenet201", 32, (6, 12, 48, 32), 64, pretrained, progress, **kwargs)


def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet("densenet161", 48, (6, 12, 36, 24), 96, pretrained, progress, **kwargs)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.ones(1, 3, 224, 224).to(device)
    model = densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    model.to(device)

    model.eval()
    with torch.inference_mode():
        y = model(x)
    print(y.size()) # [1, 10]
