import os
import json

import torch
from torchvision import transforms, datasets, models
from efficientnet_v1_gelu import efficientnet_b3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from dataset import get_data


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        # -------------------------------------------#
        #   初始化混淆矩阵
        # -------------------------------------------#
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels  # 分类标签列表

    # -------------------------------------------#
    #   预测值,真实标签累加到matrix中
    # -------------------------------------------#
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    # -------------------------------------------#
    #   统计计算各项指标
    # -------------------------------------------#
    def summary(self):
        # -------------------------------------------#
        #   准确率
        #   对角线总和除以全部数据之和
        # -------------------------------------------#
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)  # np.sum 求和
        print("the model accuracy is ", acc)

        # -------------------------------------------#
        #   precision, recall, specificity
        #   精确率,    召回率,  特异度
        # -------------------------------------------#
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]  # 对角线为 TP
            FP = np.sum(self.matrix[i, :]) - TP  # 行总数 - TP = FP  N判断为P
            FN = np.sum(self.matrix[:, i]) - TP  # 列总数 - TP = FN  P判断为N
            TN = (
                np.sum(self.matrix) - TP - FP - FN
            )  # 总和 - TP - FP - FN = TN 判断正确的N
            Precision = (
                round(TP / (TP + FP), 3) if TP + FP != 0 else 0.0
            )  # 精确率,查的准不准 round(..., 3)  保留3位小数
            Recall = (
                round(TP / (TP + FN), 3) if TP + FN != 0 else 0.0
            )  # 召回率,查的全不全
            Specificity = (
                round(TN / (TN + FP), 3) if TN + FP != 0 else 0.0
            )  # 特异度,Negative的召回率
            table.add_row(
                [self.labels[i], Precision, Recall, Specificity]
            )  # 添加到table中 参数1是标签
        print(table)

    # -------------------------------------------#
    #   绘制混淆矩阵
    # -------------------------------------------#
    def plot(self):
        matrix = self.matrix
        print(matrix)

        plt.figure(figsize=(40, 40), dpi=100)
        # -------------------------------------------#
        # 展示混淆矩阵,颜色设置为从白色到蓝色
        # -------------------------------------------#
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示右侧colorbar
        plt.colorbar()
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Confusion matrix")

        # -------------------------------------------#
        #   在图中标注数量/概率信息
        #   注意: x从左到右 y从上到下
        # -------------------------------------------#
        thresh = matrix.max() / 2  # 阈值,设置颜色
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # -------------------------------------------#
                #   注意这里的matrix[y, x]不是matrix[x, y]
                #   横坐标是x,纵坐标是y, 每一列的是x,每一行是y
                # -------------------------------------------#
                info = int(matrix[y, x])
                plt.text(
                    x,
                    y,
                    info,
                    verticalalignment="center",  # 数值在中心
                    horizontalalignment="center",
                    color="white" if info > thresh else "black",
                )  # 大于阈值就是白色文字,否则是黑色
        plt.tight_layout()  # 图形显示更紧凑

        # 解决中文显示问题
        plt.rcParams["font.sans-serif"] = ["KaiTi"]  # 指定默认字体
        plt.rcParams["axes.unicode_minus"] = (
            False  # 解决保存图像是负号'-'显示为方块的问题
        )

        plt.savefig("ConfusionMatrix.jpg")
        # 3.显示图像
        plt.show()


# -------------------------------------------#
#   计算并绘图
# -------------------------------------------#
def draw(model_path, num_classes):
    """
    model_path: 模型路径
    num_classes: 分类数
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------#
    #   验证集
    # -------------------------------------------#
    _, dataloaders, dataset_sizes = get_data(batch_size=8)
    print("验证集数量: ", dataset_sizes["valid"])
    validate_loader = dataloaders["valid"]

    # -------------------------------------------#
    #   模型
    # -------------------------------------------#
    # net = models.efficientnet_b0(num_classes=num_classes)
    net = efficientnet_b3(num_classes=num_classes)
    # load pretrain weights
    model_weight_path = model_path
    assert os.path.exists(model_path), "cannot find {} file".format(model_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # -------------------------------------------#
    #   获取标签
    # -------------------------------------------#
    json_label_path = "./class_indices.json"
    assert os.path.exists(json_label_path), "cannot find {} file".format(
        json_label_path
    )
    json_file = open(json_label_path, "r", encoding="utf-8")
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]

    # -------------------------------------------#
    #   推理并计算混淆矩阵
    # -------------------------------------------#
    confusion = ConfusionMatrix(num_classes=50, labels=labels)
    net.eval()
    with torch.inference_mode():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)  # 所有值都到0~1之间
            outputs = torch.argmax(outputs, dim=1)  # 找最大值下标
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()


if __name__ == "__main__":
    model_path = r"C:\Ai\Garbage50\训练结果\求损失平均值\更新数据集\efficientnet_v1_b3_pt_sche_gelu\models\efficientnet_v1_b3_pt_gelu_best_model.pkl"
    draw(model_path, num_classes=50)
