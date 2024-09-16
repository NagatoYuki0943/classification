import os
import glob
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from model import resnet50


def main():
    data_root = os.path.abspath(
        os.path.join(os.getcwd(), "../..")
    )  # get data root path
    image_path = os.path.join(
        data_root, "data_set", "flower_data"
    )  # flower data set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    im_height = 224
    im_width = 224
    batch_size = 16
    epochs = 20
    num_classes = 5

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    def pre_function(img):
        """
        图片均值化
        """
        # img = im.open('test.jpg')
        # img = np.array(img).astype(np.float32)

        #     红绿蓝均值
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

        return img

    # 图片生成器
    train_image_generator = ImageDataGenerator(
        horizontal_flip=True, preprocessing_function=pre_function
    )  # preprocessing_function 图片均值化

    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    train_data_gen = train_image_generator.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        shuffle=True,
        target_size=(im_height, im_width),
        class_mode="categorical",
    )
    # 图片总数
    total_train = train_data_gen.n

    # class_to_dix
    class_indices = train_data_gen.class_indices

    # idx_to_class
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # 写入文件
    json_str = json.dumps(inverse_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    # data生成器
    val_data_gen = validation_image_generator.flow_from_directory(
        directory=validation_dir,
        batch_size=batch_size,
        shuffle=False,
        target_size=(im_height, im_width),
        class_mode="categorical",
    )
    # img, _ = next(train_data_gen)
    total_val = val_data_gen.n
    print(
        "using {} images for training, {} images for validation.".format(
            total_train, total_val
        )
    )

    # 模型                            不使用平均池化层
    feature = resnet50(num_classes=5, include_top=False)
    # feature.build((None, 224, 224, 3))  # when using subclass model

    # 直接下载我转好的权重
    # download weights 链接: https://pan.baidu.com/s/1tLe9ahTMIwQAX7do_S59Zg  密码: u199
    pre_weights_path = "./pretrain_weights.ckpt"
    assert len(glob.glob(pre_weights_path + "*")), "cannot find {}".format(
        pre_weights_path
    )
    # 载入预训练模型
    feature.load_weights(pre_weights_path)

    # 冻结参数,没法训练
    feature.trainable = False
    feature.summary()

    # 重新添加分类层,只训练这两个dense层
    model = tf.keras.Sequential(
        [
            feature,
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(num_classes),
            tf.keras.layers.Softmax(),
        ]
    )
    # model.build((None, 224, 224, 3))
    # 查看模型信息,就是打印
    model.summary()

    # 损失函数有和优化器,
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    # 统计数据
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

    val_loss = tf.keras.metrics.Mean(name="val_loss")
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")

    # 计算损失,反向传播偷渡
    @tf.function
    def train_step(images, labels):
        # 要手动跟踪所有的可训练参数
        with tf.GradientTape() as tape:
            # 获取预测
            output = model(images, training=True)  # training=True
            # 计算损失
            loss = loss_object(labels, output)
        # 将损失反向传播到每一个可训练变量中
        gradients = tape.gradient(
            loss, model.trainable_variables
        )  # 只更新可以训练的参数
        # 将节点误差梯度用于更新到每个节点的参数值
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )  # 只更新可以训练的参数

        # 计算累加
        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def val_step(images, labels):
        # 不需要更新梯度
        output = model(images, training=False)  # training=True
        loss = loss_object(labels, output)

        # 计算累加
        val_loss(loss)
        val_accuracy(labels, output)

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(range(total_train // batch_size))
        for step in train_bar:
            images, labels = next(train_data_gen)
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(
                epoch + 1, epochs, train_loss.result(), train_accuracy.result()
            )

        # validate
        val_bar = tqdm(range(total_val // batch_size))
        for step in val_bar:
            test_images, test_labels = next(val_data_gen)
            val_step(test_images, test_labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(
                epoch + 1, epochs, val_loss.result(), val_accuracy.result()
            )

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            model.save_weights("./save_weights/resNet_50.ckpt", save_format="tf")


if __name__ == "__main__":
    main()
