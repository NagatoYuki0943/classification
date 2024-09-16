"""
面向过程写法

Keras相同的api

keras.Model            和 keras.models.Model
keras.Sequential       和 keras.models.Sequential
layers.Conv2D          和 layers.Convolution2D
layers.AvgPool2D       和 layers.AveragePooling2D
layers.MaxPool2D       和 layers.MaxPooling2D
layers.GlobalAvgPool2D 和 layers.GlobalAveragePooling2D  pool + flatten  平均池化和展平
layers.GlobalMaxPool2D 和 layers.GlobalMaxPooling2D      pool + flatten  最大池化和展平
"""

from tensorflow.keras import layers, Model, Sequential


class BasicBlock(layers.Layer):
    expansion = 1  # 最终输出扩展维度比例

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super().__init__(**kwargs)

        """
        注意: 顺序 conv => bn => relu 所以Conv2D的参数不设置激活函数,要手动调用
        """
        self.conv1 = layers.Conv2D(
            out_channel, kernel_size=3, strides=strides, padding="SAME", use_bias=False
        )  # 使用BatchNormalization就让bias为False
        self.bn1 = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5
        )  # keras的BN不需要写channels
        # -----------------------------------------
        self.conv2 = layers.Conv2D(
            out_channel, kernel_size=3, strides=1, padding="SAME", use_bias=False
        )
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        # 下采样层
        self.downsample = downsample
        self.relu = layers.ReLU()

        # 相加要使用相关函数
        self.add = layers.Add()

    def call(self, inputs, training=False):
        """
        training: 是否使用训练
        """

        identity = inputs
        # 下采样层
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)  # 要设置training,训练和验证状态不同
        x = self.relu(x)  # 手动调用激活函数

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # 相加要使用相关函数
        x = self.add([identity, x])
        x = self.relu(x)

        return x


class Bottleneck(layers.Layer):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """

    expansion = 4  # 最终输出扩展维度比例

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super().__init__(**kwargs)

        """
        注意: 顺序 conv => bn => relu 所以Conv2D的参数不设置激活函数,要手动调用
        """
        self.conv1 = layers.Conv2D(
            out_channel, kernel_size=1, use_bias=False, name="conv1"
        )  # name: 迁移学习时会用到
        self.bn1 = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm"
        )
        # -----------------------------------------
        self.conv2 = layers.Conv2D(
            out_channel,
            kernel_size=3,
            use_bias=False,
            strides=strides,
            padding="SAME",
            name="conv2",
        )
        self.bn2 = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm"
        )
        # -----------------------------------------
        # out_channel 发生变化
        self.conv3 = layers.Conv2D(
            out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3"
        )
        self.bn3 = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm"
        )
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample

        # 相加要使用相关函数
        self.add = layers.Add()

    def call(self, inputs, training=False):
        """
        training: 是否使用训练
        """

        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)  # 要设置training,训练和验证状态不同
        x = self.relu(x)  # 手动调用激活函数

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        # 相加要使用相关函数
        x = self.add([x, identity])
        x = self.relu(x)

        return x


def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    # 步长不为1或者进出通道不相等就设置下采样层
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential(
            [
                layers.Conv2D(
                    channel * block.expansion,
                    kernel_size=1,
                    strides=strides,
                    use_bias=False,
                    name="conv1",
                ),
                layers.BatchNormalization(
                    momentum=0.9, epsilon=1.001e-5, name="BatchNorm"
                ),
            ],
            name="shortcut",
        )

    layers_list = []
    # 增加第一层,有下采样,因此特殊处理
    layers_list.append(
        block(channel, downsample=downsample, strides=strides, name="unit_1")
    )

    # 后面的层数处理方式
    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(
    block,  # BasicBlock BottleBlock
    blocks_num,  # block个数 [layer1, layer2, layer3, layer4]
    im_width=224,
    im_height=224,
    num_classes=1000,
    include_top=True,
):  # 顶层结构 平均池化和线性层
    # tensorflow中的tensor通道排序是[N, H, W, C]
    # (None, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")

    # 开始的维度
    x = layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding="SAME",
        use_bias=False,
        name="conv1",
    )(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    #                      in           out  重复次数
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    # 分类结构
    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten  平均池化和展平
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model


def resnet18(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(
        BasicBlock, [2, 2, 2, 2], im_width, im_height, num_classes, include_top
    )


def resnet34(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(
        BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes, include_top
    )


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(
        Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top
    )


def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(
        Bottleneck, [3, 4, 23, 3], im_width, im_height, num_classes, include_top
    )


def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(
        Bottleneck, [3, 8, 36, 3], im_width, im_height, num_classes, include_top
    )
