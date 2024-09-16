"""
pytorch输入图片维度: [b, c, h, w]
tf输入图片维度:      [b, h, w, c]

model.predict(img)
"""

import os
import json

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import AlexNet_v1, AlexNet_v2


def main():
    im_height = 224
    im_width = 224

    # 载入图片
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # resize image to 224x224
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # scaling pixel value to (0-1) 变为0~1
    img = np.array(img) / 255.0

    # 添加batch维度, [h, w, c] => [0, h, w, c]
    img = np.expand_dims(img, 0)

    # 获取id_to_class
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # 创建模型并载入数据
    model = AlexNet_v1(num_classes=5)
    weighs_path = "./save_weights/myAlex.h5"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(weighs_path)
    model.load_weights(weighs_path)

    # 预测并只要第一个数据(只有一张图片)
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_class)], result[predict_class]
    )
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == "__main__":
    main()
