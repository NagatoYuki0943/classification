from tensorflow.keras import layers, Model, Sequential

# 卷积核初始化
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

# 线性层初始化
DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def VGG(feature, im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道排序是 [N H W C]
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")

    # 特征提取层
    x = feature(input_image)

    # 分类层
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    x = layers.Dense(num_classes, kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    output = layers.Softmax()(x)

    model = Model(inputs=input_image, outputs=output)
    return model


def make_feature(cfg):
    '''
    创建分类层
    '''
    feature_layers = []
    for v in cfg:
        if v == "M":
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            #                     v指的是out_channels
            conv2d = layers.Conv2D(v, kernel_size=3, padding="SAME", activation="relu", kernel_initializer=CONV_KERNEL_INITIALIZER)
            feature_layers.append(conv2d)
    # Sequential直接放列表,pytorch中的Sequential要放 *[] 或者 OrderDict
    # 返回一个 keras.Model
    return Sequential(feature_layers, name="feature")


# 配置
# 数字: 卷积层,数字指的是out_channels
# M: max_pool
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", im_height=224, im_width=224, num_classes=1000):
    assert model_name in cfgs.keys(), "not support model {}".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(make_feature(cfg), im_height=im_height, im_width=im_width, num_classes=num_classes)
    return model
