"""
损失函数:
    CategoricalCrossentropy()分类使用的one_hot,使用单独的数字要使用SparseCategoricalCrossentropy()分类器

"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os


def main():
    # 图像位置
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

    # 保存权重的位置
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    # 基本参数
    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10

    # 图像data生成器,可以载入文件夹的图像数据和调整图像
    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(
        rescale=1.0 / 255,  # 缩放 0~255 -> 0~1
        horizontal_flip=True,
    )  # 水平翻转
    validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)

    # 训练集
    train_data_gen = train_image_generator.flow_from_directory(
        directory=train_dir,  # 路径里面有很多的文件夹,每一个文件夹是一个分类,和ImageFolder相同
        batch_size=batch_size,
        shuffle=True,
        target_size=(im_height, im_width),  # 输出尺寸大小
        class_mode="categorical",
    )  # 分类
    # 总数
    total_train = train_data_gen.n

    # 分类与id关系 class_to_idx
    class_indices = train_data_gen.class_indices  # {'daisy':0, 'dandelion': 1..}

    # class id 互换
    inverse_dict = dict(
        (val, key) for key, val in class_indices.items()
    )  # {0: 'daisy', 0: 'dandelion'..}
    # 写入文件
    json_str = json.dumps(inverse_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    # 验证集
    val_data_gen = validation_image_generator.flow_from_directory(
        directory=validation_dir,
        batch_size=batch_size,
        shuffle=False,
        target_size=(im_height, im_width),
        class_mode="categorical",
    )
    total_val = val_data_gen.n
    print(
        "using {} images for training, {} images for validation.".format(
            total_train, total_val
        )
    )

    # 返回 图片数据和label, label是one_hot模式
    # sample_training_images, sample_training_labels = next(train_data_gen)  # label is one-hot coding
    #
    # # This function will plot images in the form of a grid with 1 row
    # # and 5 columns where images are placed in each column.
    # def plotImages(images_arr):
    #     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    #     axes = axes.flatten()
    #     for img, ax in zip(images_arr, axes):
    #         ax.imshow(img)
    #         ax.axis('off')
    #     plt.tight_layout()
    #     plt.show()
    #
    # plotImages(sample_training_images[:5])

    # 实例化模型
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)

    # model = AlexNet_v2(class_num=5)
    # 使用模型的方法,要使用build才能使用summary()
    # model.build((batch_size, 224, 224, 3))  # when using subclass model

    # 查看模型信息,就是打印
    model.summary()

    # 编译,设置优化器,损失
    # CategoricalCrossentropy()分类使用的one_hot,使用单独的数字要使用SparseCategoricalCrossentropy()分类器
    # from_logits 模型使用softmax,这里就设置为False,否则就设置为True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )  # metrics是监控的信息,这里选择的准确率

    # 回调函数列表, 这里使用的保存模型
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./save_weights/myAlex.h5",
            save_best_only=True,  # 只保存最佳的模型
            save_weights_only=True,  # 只保存权重文件
            monitor="val_loss",
        )
    ]  # 通过loss作为评估

    # 训练
    history = model.fit(
        x=train_data_gen,  # 训练集
        steps_per_epoch=total_train
        // batch_size,  # 每一轮迭代多少次 一个epoch有多少个batch
        epochs=epochs,
        validation_data=val_data_gen,  # 验证集
        validation_steps=total_val
        // batch_size,  # 每一轮迭代多少次 一个epoch有多少个batch
        callbacks=callbacks,
    )  # 回调函数

    # 训练完的信息,返回字典
    history_dict = history.history
    train_loss = history_dict["loss"]  # 训练损失
    train_accuracy = history_dict["accuracy"]  # 训练准确率
    val_loss = history_dict["val_loss"]  # 验证集损失
    val_accuracy = history_dict["val_accuracy"]  # 验证集准确率

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label="train_loss")
    plt.plot(range(epochs), val_loss, label="val_loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label="train_accuracy")
    plt.plot(range(epochs), val_accuracy, label="val_accuracy")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.show()

    # history = model.fit_generator(generator=train_data_gen,
    #                               steps_per_epoch=total_train // batch_size,
    #                               epochs=epochs,
    #                               validation_data=val_data_gen,
    #                               validation_steps=total_val // batch_size,
    #                               callbacks=callbacks)

    # TF底层API
    # # using keras low level api for training
    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    #
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    #
    #
    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)
    #
    #
    # @tf.function
    # def test_step(images, labels):
    #     predictions = model(images, training=False)
    #     t_loss = loss_object(labels, predictions)
    #
    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)
    #
    #
    # best_test_loss = float('inf')
    # for epoch in range(1, epochs+1):
    #     train_loss.reset_states()        # clear history info
    #     train_accuracy.reset_states()    # clear history info
    #     test_loss.reset_states()         # clear history info
    #     test_accuracy.reset_states()     # clear history info
    #     for step in range(total_train // batch_size):
    #         images, labels = next(train_data_gen)
    #         train_step(images, labels)
    #
    #     for step in range(total_val // batch_size):
    #         test_images, test_labels = next(val_data_gen)
    #         test_step(test_images, test_labels)
    #
    #     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    #     print(template.format(epoch,
    #                           train_loss.result(),
    #                           train_accuracy.result() * 100,
    #                           test_loss.result(),
    #                           test_accuracy.result() * 100))
    #     保存模型
    #     if test_loss.result() < best_test_loss:
    #        model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')


if __name__ == "__main__":
    main()
