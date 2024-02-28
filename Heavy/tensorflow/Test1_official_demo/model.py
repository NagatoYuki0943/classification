'''
tf中Dense,Conv2D不用给定输入的节点个数,只需要给出输出的个数即可

Conv2D参数:
    padding = 'valid' / 'same'
    data_format = 'channels_last' / 'channels_first'
        channels_last:  [b, h, w, c]
        channels_first: [b, c, h, w]

https://www.tensorflow.org/tutorials/quickstart/advanced
'''


from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class MyModel(Model):
    def __init__(self):
        super().__init__()
        #                  out, k  激活
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        #               out个数
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)      # input[batch, 28, 28, 1] output[batch, 26, 26, 32]
        x = self.flatten(x)    # output [batch, 21632]
        x = self.d1(x)         # output [batch, 128]
        return self.d2(x)      # output [batch, 10]
