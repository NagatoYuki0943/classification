import tensorflow as tf
from tensorflow.keras import layers, Model


class ConvBNReLU(layers.Layer):
    '''
    Conv + BN + ReLU
    '''
    def __init__(self,
                 filters: int = 1,
                 kernel_size: int = 1,
                 strides: int = 1,
                 padding: str = 'same',
                 **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  use_bias=False,       # 使用BN不使用bias
                                  kernel_regularizer=tf.keras.regularizers.l2(4e-5),    # 正则项
                                  name="conv1")
        self.bn = layers.BatchNormalization(momentum=0.9, name="bn")
        self.relu = layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)   # 训练or验证
        x = self.relu(x)
        return x


class DWConvBN(layers.Layer):
    '''
    DWConv + BN 没有ReLU
    '''
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 padding: str = 'same',
                 **kwargs):
        super().__init__(**kwargs)
        # DepthwiseConv2D不需要定义channel
        self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              use_bias=False,
                                              kernel_regularizer=tf.keras.regularizers.l2(4e-5), # 正则项
                                              name="dw1")
        self.bn = layers.BatchNormalization(momentum=0.9, name="bn")

    def call(self, inputs, training=None, **kwargs):
        x = self.dw_conv(inputs)
        x = self.bn(x, training=training)   # 训练or验证
        return x


class ChannelShuffle(layers.Layer):
    '''
    混淆通道
    '''
    def __init__(self, shape,   # 传入特征矩阵的shape [b, h, w, c]
                        groups: int = 2, **kwargs):
        super().__init__(**kwargs)
        batch_size, height, width, num_channels = shape
        # 必须是2的整数倍,不然报错
        assert num_channels % 2 == 0
        # 每个组合的通道个数
        channel_per_group = num_channels // groups

        # Tuple of integers, does not include the samples dimension (batch size).
        # 不需要传入batch维度
        # [batch, height, width, channcels] => [height, width, groups, channel_per_group]
        self.reshape1 = layers.Reshape((height, width, groups, channel_per_group))

        # [height, width, groups, channel_per_group] => [batch, height, width, channcels]
        self.reshape2 = layers.Reshape((height, width, num_channels))

    def call(self, inputs, **kwargs):
        x = self.reshape1(inputs)
        # 最后两个维度顺序变化
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = self.reshape2(x)
        return x


class ChannelSplit(layers.Layer):
    '''
    将输入通道一分为二, 只有没有捷径分支的情况才会使用(stride=1)
    pytorch中使用了 x.chunk(2, dim=1) 直接分为了两份
    '''
    def __init__(self, num_splits: int = 2, # 分为几份
                        **kwargs):
        super().__init__(**kwargs)
        self.num_splits = num_splits

    def call(self, inputs, **kwargs):
        b1, b2 = tf.split(inputs,
                          num_or_size_splits=self.num_splits,   # 分为几份
                          axis=-1)                              # channel维度划分
        return b1, b2


def shuffle_block_s1(inputs,        # 特征矩阵data
                    output_c: int,  # out_channel
                    stride: int, prefix: str):
    '''
    stride = 1
    in_channel = out_channel
    in_channel 一分为二
    '''
    if stride != 1:
        raise ValueError("illegal stride value.")

    assert output_c % 2 == 0
    # 每个分支channel
    branch_c = output_c // 2

    # in_channel 一分为二
    x1, x2 = ChannelSplit(name=prefix + "/split")(inputs)

    # main branch
    # 1x1
    x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv1")(x2)
    # 3x3DWConv
    x2 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b2_dw1")(x2)
    # 1x1
    x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv2")(x2)

    # 拼接
    x = layers.Concatenate(name=prefix + "/concat")([x1, x2])
    # 混淆
    x = ChannelShuffle(x.shape, name=prefix + "/channelshuffle")(x)

    return x


def shuffle_block_s2(inputs,        # 特征矩阵data
                    output_c: int,  # out_channel
                    stride: int, prefix: str):
    '''
    stride = 2
    in_channel = out_channel
    in_channel不划分,计算时channel减半
    '''
    if stride != 2:
        raise ValueError("illegal stride value.")

    assert output_c % 2 == 0
    branch_c = output_c // 2

    # shortcut branch
    # 3x3DWConv
    x1 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b1_dw1")(inputs)
    x1 = ConvBNReLU(filters=branch_c, name=prefix + "/b1_conv1")(x1)

    # main branch
    # 1x1
    x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv1")(inputs)
    # 3x3DWConv
    x2 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b2_dw1")(x2)
    # 1x1
    x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv2")(x2)

    # 拼接
    x = layers.Concatenate(name=prefix + "/concat")([x1, x2])
    # 混淆
    x = ChannelShuffle(x.shape, name=prefix + "/channelshuffle")(x)

    return x


def shufflenet_v2(num_classes: int,
                  input_shape: tuple,           # (224, 224, 3)
                  stages_repeats: list,         # stage2,3,4重复次数: [4, 8, 4]
                  stages_out_channels: list):   # conv1,stage2,3,4,conv5输出维度: [24, 116, 232, 464, 1024]
    img_input = layers.Input(shape=input_shape)

    # 确定长度
    if len(stages_repeats) != 3:
        raise ValueError("expected stages_repeats as list of 3 positive ints")
    if len(stages_out_channels) != 5:
        raise ValueError("expected stages_out_channels as list of 5 positive ints")

    # 开始卷积
    x = ConvBNReLU(filters=stages_out_channels[0],
                   kernel_size=3,
                   strides=2,
                   name="conv1")(img_input)
    # 池化
    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2,
                            padding='same',
                            name="maxpool")(x)

    # stage2, 3, 4
    stage_name = ["stage{}".format(i) for i in [2, 3, 4]]
    for name, repeats, output_channels in zip(stage_name,
                                              stages_repeats,           # 有3个数据,循环3次
                                              stages_out_channels[1:]): # 第一个已经使用了,略过, 循环3个数据,不过由于上面两个参数只有3个,所以最后一个用不到
        # 每个stage循环stage次
        for i in range(repeats):
            # 第0次改变channel和HW,其他不变(channel变化是相对于之前的数据)
            if i == 0:
                x = shuffle_block_s2(x, output_c=output_channels, stride=2, prefix=name + "_{}".format(i))
            else:
                x = shuffle_block_s1(x, output_c=output_channels, stride=1, prefix=name + "_{}".format(i))

    # conv5 ,out_channel是stages_out_channels最后一个
    x = ConvBNReLU(filters=stages_out_channels[-1], name="conv5")(x)

    # [b, h, w, c] => [b, c]
    x = layers.GlobalAveragePooling2D(name="globalpool")(x)

    x = layers.Dense(units=num_classes, name="fc")(x)
    x = layers.Softmax()(x)

    model = Model(img_input, x, name="ShuffleNetV2_1.0")

    return model


def shufflenet_v2_x0_5(num_classes=1000, input_shape=(224, 224, 3)):
    model = shufflenet_v2(num_classes=num_classes,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channels=[24, 48, 96, 192, 1024])
    return model


def shufflenet_v2_x1_0(num_classes=1000, input_shape=(224, 224, 3)):
    # 权重链接: https://pan.baidu.com/s/1M2mp98Si9eT9qT436DcdOw  密码: mhts
    model = shufflenet_v2(num_classes=num_classes,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channels=[24, 116, 232, 464, 1024])
    return model


def shufflenet_v2_x2_0(num_classes=1000, input_shape=(224, 224, 3)):
    model = shufflenet_v2(num_classes=num_classes,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channels=[24, 244, 488, 976, 2048])
    return model
