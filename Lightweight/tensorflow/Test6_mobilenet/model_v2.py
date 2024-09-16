"""
keras中有专门的DW卷积
不需要写维度
layers.DepthwiseConv2D(kernel_size=3, padding='SAME', strides=stride, use_bias=False)

"""

from tensorflow.keras import layers, Model, Sequential


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


class ConvBNReLU(layers.Layer):
    """
    卷积层 = 卷积+ BN + ReLU6
    不负责DW卷积,keras自带了DW卷积
    """

    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            filters=out_channel,
            kernel_size=kernel_size,
            strides=stride,
            padding="SAME",
            use_bias=False,
            name="Conv2d",
        )
        self.bn = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name="BatchNorm"
        )
        self.activation = layers.ReLU(max_value=6.0)  # ReLU6

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class InvertedResidual(layers.Layer):
    """
        倒残差结构
        残差:   两端channel多,中间channel少
    ​       降维 --> 升维
        倒残差: 两端channel少,中间channel多
    ​       升维 --> 降维
    """

    def __init__(self, in_channel, out_channel, stride, expand_ratio, **kwargs):
        """
        expand_ratio: 扩展因子,表格中的t

        """
        super().__init__(**kwargs)

        # 第一层卷积核个数,第一层输出维度
        self.hidden_channel = in_channel * expand_ratio

        # 判断是否使用捷径分支 只有当stride=1且n_channel == out_channel才使用
        self.use_shortcut = (stride == 1) and (in_channel == out_channel)

        layer_list = []
        # 扩展因子是否为1,第一层为1就不需要使用第一个 1x1 的卷积层
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layer_list.append(
                ConvBNReLU(
                    out_channel=self.hidden_channel, kernel_size=1, name="expand"
                )
            )

        layer_list.extend(
            [
                # 3x3 depthwise conv  DW卷积,in_channel = out_channel = gropus
                layers.DepthwiseConv2D(
                    kernel_size=3,
                    padding="SAME",
                    strides=stride,
                    use_bias=False,
                    name="depthwise",
                ),
                layers.BatchNormalization(
                    momentum=0.9, epsilon=1e-5, name="depthwise/BatchNorm"
                ),
                layers.ReLU(max_value=6.0),
                # 1x1 pointwise conv(linear) PW卷积,变换维度,不使用激活函数,就是线性激活
                layers.Conv2D(
                    filters=out_channel,
                    kernel_size=1,
                    strides=1,
                    padding="SAME",
                    use_bias=False,
                    name="project",
                ),
                layers.BatchNormalization(
                    momentum=0.9, epsilon=1e-5, name="project/BatchNorm"
                ),
            ]
        )
        self.main_branch = Sequential(layer_list, name="expanded_conv")

    def call(self, inputs, training=False, **kwargs):
        if self.use_shortcut:
            return inputs + self.main_branch(inputs, training=training)
        else:
            return self.main_branch(inputs, training=training)


def MobileNetV2(
    im_height=224,
    im_width=224,
    num_classes=1000,
    alpha=1.0,  # 调整卷积核个数参数,默认为1
    round_nearest=8,  # 调整为8的倍数
    include_top=True,
):
    block = InvertedResidual
    # 第一层输入的个数   将卷积核个数(输出通道个数)调整为round_nearest的整数倍 就是调整为8个整数倍
    input_channel = _make_divisible(32 * alpha, round_nearest)
    # 最后通道个数
    last_channel = _make_divisible(1280 * alpha, round_nearest)
    inverted_residual_setting = [
        # t 扩展因子,第一层卷积让channel维度变多
        # c out_channel 或 k1
        # n bottlenet 重复次数
        # s 步长(每一个block的第一层步长)
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")

    # conv1 layer 第一层卷积     3  32*alpha
    x = ConvBNReLU(input_channel, stride=2, name="Conv")(input_image)

    # 一系列block
    for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
        output_channel = _make_divisible(c * alpha, round_nearest)
        # 添加n次
        for i in range(n):
            # stride只有第一次为2,其余的为1
            stride = s if i == 0 else 1
            #         x的输出维度 [b, h, w, c] 最后一个是维度
            x = block(x.shape[-1], output_channel, stride, expand_ratio=t)(x)

    # 输出层
    x = ConvBNReLU(last_channel, kernel_size=1, name="Conv_1")(x)

    # 分类层
    if include_top is True:
        # building classifier
        x = layers.GlobalAveragePooling2D()(x)  # pool + flatten
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(num_classes, name="Logits")(x)
    else:
        output = x

    model = Model(inputs=input_image, outputs=output)
    return model
