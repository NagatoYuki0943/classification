"""
pytorch会自动跟踪所有的可训练参数
tf不会,要手动跟踪所有的可训练参数
    with tf.GradientTape() as tape:
        ...
    ...

损失函数:
    CategoricalCrossentropy()分类使用的one_hot,使用单独的数字要使用SparseCategoricalCrossentropy()分类器

https://www.tensorflow.org/tutorials/quickstart/advanced
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from model import MyModel


def main():
    # 数据集
    mnist = tf.keras.datasets.mnist

    # 获取训练和测试数据
    # x_train: [b, h, w]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 归一化
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    # [b, h, w] => [b, h, w, 1]
    x_train = x_train[..., tf.newaxis]  # tf.newaxis 一个维度
    x_test = x_test[..., tf.newaxis]

    # 数据生成器                                  img和label合并成元组  随机
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # create model
    model = MyModel()

    # 损失函数    CategoricalCrossentropy()分类使用的one_hot,使用单独的数字要使用SparseCategoricalCrossentropy()分类器
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # 优化器
    optimizer = tf.keras.optimizers.Adam()

    # 用于统计和训练过程的损失和准确率变化
    # define train_loss and train_accuracy
    train_loss = tf.keras.metrics.Mean(name="train_loss")  # Mean 平均值
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    # define test_loss and test_accuracy
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    # 计算损失,反向传播偷渡
    # define train function including calculating loss, applying gradient and calculating accuracy
    @tf.function
    def train_step(images, labels):
        # 要手动跟踪所有的可训练参数
        with tf.GradientTape() as tape:
            # 获取预测
            predictions = model(images)
            # 计算损失
            loss = loss_object(labels, predictions)
        # 将损失反向传播到每一个可训练变量中
        gradients = tape.gradient(loss, model.trainable_variables)
        # 将节点误差梯度用于更新到每个节点的参数值
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # 计算累加
        train_loss(loss)
        train_accuracy(labels, predictions)

    # define test function including calculating loss and calculating accuracy
    @tf.function
    def test_step(images, labels):
        # 不需要更新梯度
        predictions = model(images)
        t_loss = loss_object(labels, predictions)

        # 计算累加
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        # 计数器每次都清空
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        test_loss.reset_states()  # clear history info
        test_accuracy.reset_states()  # clear history info

        # 遍历训练迭代器
        for images, labels in train_ds:
            train_step(images, labels)
        # 遍历测试迭代器
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )


if __name__ == "__main__":
    main()
