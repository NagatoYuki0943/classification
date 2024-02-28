'''
pytorch输入图片维度: [b, c, h, w]
tf输入图片维度:      [b, h, w, c]
'''


from tensorflow.keras import layers, Model, Sequential


def AlexNet_v1(im_height=224, im_width=224, num_classes=1000):
    # tensorflow中的tensor通道排序是 [N H W C]
    # 图像输入
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # output(N, 224, 224, 3)

    # 手动padding (1, 2), (1, 2) 前面 上下左右
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # output(N, 227, 227, 3)

    #                 out_channel 不需要写 in_channel
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(N, 55, 55, 48)
    #                    高宽减少到不要一半
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(N, 27, 27, 48)
    x = layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(N, 27, 27, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(N, 13, 13, 128)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(N, 13, 13, 192)
    x = layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(N, 13, 13, 192)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(N, 13, 13, 128)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(N, 6, 6, 128)

    # (N, 6, 6, 128) => (N, 6*6*128)
    x = layers.Flatten()(x)                         # output(N, 6*6*128)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)    # output(N, 2048)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)    # output(N, 2048)
    x = layers.Dense(num_classes)(x)                # output(N, 5)
    # softmax处理
    predict = layers.Softmax()(x)

    # 最后通过Model创建模型
    model = Model(inputs=input_image, outputs=predict)
    return model


# 类似于pytorch中的写法
class AlexNet_v2(Model):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),                                 # output(N, 227, 227, 3)
            layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),        # output(N, 55, 55, 48)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(N, 27, 27, 48)
            layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),   # output(N, 27, 27, 128)
            layers.MaxPool2D(pool_size=3, strides=2),                               # output(N, 13, 13, 128)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(N, 13, 13, 192)
            layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(N, 13, 13, 192)
            layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),   # output(N, 13, 13, 128)
            layers.MaxPool2D(pool_size=3, strides=2)])                              # output(N, 6, 6, 128)

        self.flatten = layers.Flatten()
        self.classifier = Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation="relu"),                                  # output(N, 2048)
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),                                   # output(N, 2048)
            layers.Dense(num_classes),                                              # output(N, 5)
            layers.Softmax()
        ])

    def call(self, inputs, **kwargs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
