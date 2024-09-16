from typing import Union
from functools import partial
from tensorflow.keras import layers, Model


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    将卷积核个数(输出通道个数)调整为最接近round_nearest的整数倍,就是8的整数倍,对硬件更加友好
    ch:      输出通道个数
    divisor: 奇数,必须将ch调整为它的整数倍
    min_ch:  最小通道数

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def correct_pad(input_size: Union[int, tuple], kernel_size: int):
    """
    使用卷积 stride=2时计算合适的padding
    Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments:
      input_size: Input tensor size.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      A tuple.
    """

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    kernel_size = (kernel_size, kernel_size)

    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]))


class HardSigmoid(layers.Layer):
    """
    HardSigmoid激活函数
    HardSigmoid = ReLU(x + 3) / 6
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relu6 = layers.ReLU(6.0)

    def call(self, inputs, **kwargs):
        x = self.relu6(inputs + 3) * (1.0 / 6)
        return x


class HardSwish(layers.Layer):
    """
    HardSwish激活函数
    HardSwish(x) = x * HardSigmoid(x)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hard_sigmoid = HardSigmoid()

    def call(self, inputs, **kwargs):
        x = self.hard_sigmoid(inputs) * inputs
        return x


def _se_block(
    inputs,  # 输入数据 [batch, height, width, channel]
    filters,  # 输入数据的channel
    prefix,  # Conv2D名字前缀
    se_ratio=1 / 4.0,
):  # filters缩小倍率
    """
    注意力机制
    """
    # [batch, height, width, channel] -> [batch, channel]
    x = layers.GlobalAveragePooling2D(name=prefix + "squeeze_excite/AvgPool")(inputs)

    # Target shape. Tuple of integers, does not include the samples dimension (batch size).
    # [batch, channel] -> [batch, 1, 1, channel]  这样才能相乘
    x = layers.Reshape((1, 1, filters))(x)

    # fc1 filters => filters / 4
    x = layers.Conv2D(
        filters=_make_divisible(filters * se_ratio),
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv",
    )(x)
    x = layers.ReLU(name=prefix + "squeeze_excite/Relu")(x)

    # fc2 filters / 4 => filters
    x = layers.Conv2D(
        filters=filters,
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv_1",
    )(x)
    x = HardSigmoid(name=prefix + "squeeze_excite/HardSigmoid")(x)

    # 相乘
    x = layers.Multiply(name=prefix + "squeeze_excite/Mul")([inputs, x])
    return x


def _inverted_res_block(
    x,  # data
    input_c: int,  # input_channel
    kernel_size: int,  # DW卷积kennel_size
    exp_c: int,  # expanded_channel
    out_c: int,  # out_channel
    use_se: bool,  # whether using SE
    activation: str,  # RE or HS 激活函数
    stride: int,
    block_id: int,  # Bneck的索引, 0 1 2 3 4 都不相同
    alpha: float = 1.0,
):  # 调整宽度的超参数
    """
    倒残差
    """
    # 添加默认参数
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)

    input_c = _make_divisible(input_c * alpha)
    exp_c = _make_divisible(exp_c * alpha)
    out_c = _make_divisible(out_c * alpha)

    # 激活函数
    act = layers.ReLU if activation == "RE" else HardSwish

    shortcut = x
    prefix = "expanded_conv/"

    # block_id == 0 就不创建第一个 1x1Conv,因为第一个Bneck的维度不变
    if block_id:
        # expand channel
        prefix = "expanded_conv_{}/".format(block_id)
        x = layers.Conv2D(
            filters=exp_c,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = bn(name=prefix + "expand/BatchNorm")(x)
        x = act(name=prefix + "expand/" + act.__name__)(x)

    # 步长为2就使用padding
    if stride == 2:
        input_size = (x.shape[1], x.shape[2])  # height, width
        x = layers.ZeroPadding2D(
            padding=correct_pad(input_size, kernel_size), name=prefix + "depthwise/pad"
        )(x)

    # DW卷积 kernel = 5x5 or 3x3
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",  # 步长不为2就用上面的调整
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = bn(name=prefix + "depthwise/BatchNorm")(x)
    x = act(name=prefix + "depthwise/" + act.__name__)(x)

    # 注意力机制
    if use_se:
        x = _se_block(x, filters=exp_c, prefix=prefix)

    # 1x1
    x = layers.Conv2D(
        filters=out_c,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project",
    )(x)
    x = bn(name=prefix + "project/BatchNorm")(x)

    # 步长为1同时 in_channel == out_channel 才短接
    if stride == 1 and input_c == out_c:
        x = layers.Add(name=prefix + "Add")([shortcut, x])

    return x


def mobilenet_v3_large(
    input_shape=(224, 224, 3),  # 图片形状
    num_classes=1000,
    alpha=1.0,  # 超参数
    include_top=True,
):  # 分类结构
    """
    download weights url:
    链接: https://pan.baidu.com/s/13uJznKeqHkjUp72G_gxe8Q  密码: 8quu
    """
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
    img_input = layers.Input(shape=input_shape)

    # 开始的卷积
    x = layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv",
    )(img_input)
    x = bn(name="Conv/BatchNorm")(x)
    x = HardSwish(name="Conv/HardSwish")(x)

    # 创建提取层并执行
    inverted_cnf = partial(_inverted_res_block, alpha=alpha)
    # input, input_c, k_size, expand_c, out_c, use_se, activation, stride, block_id
    x = inverted_cnf(x, 16, 3, 16, 16, False, "RE", 1, 0)
    x = inverted_cnf(x, 16, 3, 64, 24, False, "RE", 2, 1)
    x = inverted_cnf(x, 24, 3, 72, 24, False, "RE", 1, 2)
    x = inverted_cnf(x, 24, 5, 72, 40, True, "RE", 2, 3)
    x = inverted_cnf(x, 40, 5, 120, 40, True, "RE", 1, 4)
    x = inverted_cnf(x, 40, 5, 120, 40, True, "RE", 1, 5)
    x = inverted_cnf(x, 40, 3, 240, 80, False, "HS", 2, 6)
    x = inverted_cnf(x, 80, 3, 200, 80, False, "HS", 1, 7)
    x = inverted_cnf(x, 80, 3, 184, 80, False, "HS", 1, 8)
    x = inverted_cnf(x, 80, 3, 184, 80, False, "HS", 1, 9)
    x = inverted_cnf(x, 80, 3, 480, 112, True, "HS", 1, 10)
    x = inverted_cnf(x, 112, 3, 672, 112, True, "HS", 1, 11)
    x = inverted_cnf(x, 112, 5, 672, 160, True, "HS", 2, 12)
    x = inverted_cnf(x, 160, 5, 960, 160, True, "HS", 1, 13)
    x = inverted_cnf(x, 160, 5, 960, 160, True, "HS", 1, 14)

    # 最后的conv
    last_c = _make_divisible(160 * 6 * alpha)
    # fc1输出
    last_point_c = _make_divisible(1280 * alpha)

    x = layers.Conv2D(
        filters=last_c, kernel_size=1, padding="same", use_bias=False, name="Conv_1"
    )(x)
    x = bn(name="Conv_1/BatchNorm")(x)
    x = HardSwish(name="Conv_1/HardSwish")(x)

    if include_top is True:
        x = layers.GlobalAveragePooling2D()(
            x
        )  # [batch, height, width, channel] -> [batch, channel]
        x = layers.Reshape((1, 1, last_c))(
            x
        )  # [batch, channel] -> [batch, 1, 1, channel]

        # fc1
        x = layers.Conv2D(
            filters=last_point_c, kernel_size=1, padding="same", name="Conv_2"
        )(x)
        x = HardSwish(name="Conv_2/HardSwish")(x)

        # fc2
        x = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding="same",
            name="Logits/Conv2d_1c_1x1",
        )(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name="Predictions")(x)

    # 创建模型
    model = Model(img_input, x, name="MobilenetV3large")

    return model


def mobilenet_v3_small(
    input_shape=(224, 224, 3), num_classes=1000, alpha=1.0, include_top=True
):
    """
    download weights url:
    链接: https://pan.baidu.com/s/1vrQ_6HdDTHL1UUAN6nSEcw  密码: rrf0
    """
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
    img_input = layers.Input(shape=input_shape)

    x = layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv",
    )(img_input)
    x = bn(name="Conv/BatchNorm")(x)
    x = HardSwish(name="Conv/HardSwish")(x)

    inverted_cnf = partial(_inverted_res_block, alpha=alpha)
    # input, input_c, k_size, expand_c, use_se, activation, stride, block_id
    x = inverted_cnf(x, 16, 3, 16, 16, True, "RE", 2, 0)
    x = inverted_cnf(x, 16, 3, 72, 24, False, "RE", 2, 1)
    x = inverted_cnf(x, 24, 3, 88, 24, False, "RE", 1, 2)
    x = inverted_cnf(x, 24, 5, 96, 40, True, "HS", 2, 3)
    x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 4)
    x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 5)
    x = inverted_cnf(x, 40, 5, 120, 48, True, "HS", 1, 6)
    x = inverted_cnf(x, 48, 5, 144, 48, True, "HS", 1, 7)
    x = inverted_cnf(x, 48, 5, 288, 96, True, "HS", 2, 8)
    x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 9)
    x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 10)

    last_c = _make_divisible(96 * 6 * alpha)
    last_point_c = _make_divisible(1024 * alpha)

    x = layers.Conv2D(
        filters=last_c, kernel_size=1, padding="same", use_bias=False, name="Conv_1"
    )(x)
    x = bn(name="Conv_1/BatchNorm")(x)
    x = HardSwish(name="Conv_1/HardSwish")(x)

    if include_top is True:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1, 1, last_c))(x)

        # fc1
        x = layers.Conv2D(
            filters=last_point_c, kernel_size=1, padding="same", name="Conv_2"
        )(x)
        x = HardSwish(name="Conv_2/HardSwish")(x)

        # fc2
        x = layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            padding="same",
            name="Logits/Conv2d_1c_1x1",
        )(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name="Predictions")(x)

    model = Model(img_input, x, name="MobilenetV3large")

    return model
