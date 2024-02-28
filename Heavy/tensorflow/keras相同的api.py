'''
面向过程写法

Keras相同的api

keras.Model            和 keras.models.Model
keras.Sequential       和 keras.models.Sequential
layers.Conv2D          和 layers.Convolution2D
layers.AvgPool2D       和 layers.AveragePooling2D
layers.MaxPool2D       和 layers.MaxPooling2D
layers.GlobalAvgPool2D 和 layers.GlobalAveragePooling2D  pool + flatten 平均池化和展平
layers.GlobalMaxPool2D 和 layers.GlobalMaxPooling2D      pool + flatten 最大池化和展平
'''

from tensorflow.keras import layers, Model, Sequential
layers.Conv2D()
layers.Convolution2D()

layers.MaxPool2D()
layers.MaxPooling2D()

layers.AvgPool2D()
layers.AveragePooling2D()

layers.GlobalAvgPool2D()
layers.GlobalAveragePooling2D()

layers.GlobalMaxPool2D()
layers.GlobalMaxPooling2D()