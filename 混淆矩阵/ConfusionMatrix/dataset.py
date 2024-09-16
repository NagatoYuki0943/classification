"""
垃圾数据集
使用pytorch自带的目录读取方式
"""

from torchvision.datasets import ImageFolder
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def get_data(
    data_dir=r"C:\Ai\\Garbage50\data\\", batch_size=8, resize=224, num_workers=0
):
    """
    获取数据集
    """

    data_transforms = {
        "train": transforms.Compose(
            [
                # transforms.Resize(resize*1.25),
                # transforms.RandomCrop((resize, resize)),
                transforms.RandomResizedCrop(
                    resize
                ),  # 随机缩小剪裁,输出为(resize, resize)
                transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1
                ),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
                transforms.ToTensor(),  # 转化为tensor,并归一化
                transforms.Normalize(
                    [0.485, 0.456, 0.406],  # 减去均值(前面),除以标准差(后面)
                    [0.229, 0.224, 0.225],
                ),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(int(resize * 1.25)),
                transforms.RandomCrop((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # 通过遍历的方式制作两个dataset,dataloader,dataset_size
    image_datasets = {
        x: ImageFolder(root=os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ["train", "valid"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        for x in ["train", "valid"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid"]}

    print(f"训练集总数：{dataset_sizes['train']} 验证集总数：{dataset_sizes['valid']}")
    return image_datasets, dataloaders, dataset_sizes


if __name__ == "__main__":
    image_datasets, dataloaders, dataset_sizes = get_data()
    # image_datasets, dataloaders, dataset_sizes = get_data('/content/Garbage50/data')

    print(dataset_sizes)
    # {'train': 46560, 'valid': 10701, 'test': 9099}    66,360
    print("*" * 50)

    # 获取文件夹名字和标签字典
    class_to_idx = image_datasets["train"].class_to_idx
    print(class_to_idx)
    # {'000 厨余垃圾 剩饭剩菜': 0, '001 厨余垃圾 骨头': 1,
    print("*" * 50)

    # 获取文件夹列表
    classes = image_datasets["train"].classes
    print(classes)
    # ['000 厨余垃圾 剩饭剩菜', '001 厨余垃圾 骨头', '002 厨余垃圾 鱼骨'
    print("*" * 50)

    # 通过id找到文件夹
    print(classes[1])
    # 003 厨余垃圾 蛋糕
    print("*" * 50)

    # with open("C:\Ai\Garbage50\data\\target2name.json", mode='r',encoding='utf-8') as f:
    #    target2name = json.load(f)

    #    print(target2name[str(10)])   # 010 厨余垃圾 面条
