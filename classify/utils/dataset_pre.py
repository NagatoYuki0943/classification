# https://pytorch.org/vision/stable/datasets.html

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import (
    v2,
)  # https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html

# 防止下载数据集报错
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def minst(
    variance="MNIST",
    batch_size=16,
    root="../datasets",
    download=True,
    resize=28,
    num_workers=8,
):
    """MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST

    Args:
        variance (str, optional):   数据集名称, [MNIST, EMNIST, FashionMNIST, KMNIST, QMNIST]. Defaults to 'DTD'.
        batch_size (int, optional): batch_size. Defaults to 16.
        root (str, optional):       存放路径. Defaults to './dataset'.
        download (bool, optional):  是否下载数据集. Defaults to True.
        resize (int, optional):     图片大小. Defaults to 28.
        num_workers (int, optional):并行数. Defaults to 8.

    Returns:
        train_dataset, val_dataset, train_dataloader, val_dataloader
    """
    assert variance in [
        "MNIST",
        "EMNIST",
        "FashionMNIST",
        "KMNIST",
        "QMNIST",
    ], "variance should in CIFAR10, EMNIST, CIFAR100, FashionMNIST, KMNIST and QMNIST"

    dataset = {
        "MNIST": datasets.MNIST,
        "EMNIST": datasets.EMNIST,
        "FashionMNIST": datasets.FashionMNIST,
        "KMNIST": datasets.KMNIST,
        "QMNIST": datasets.QMNIST,
    }[variance]

    transform = v2.Compose(
        [
            v2.Resize(resize),
            transforms.ToTensor(),
            # v2.ToDtype(torch.float32, scale=True),  # 不会将PIL转换为Tensor
            # v2.Normalize([0.485], [0.229]),
        ]
    )

    train_dataset = dataset(root, train=True, transform=transform, download=download)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_dataset = dataset(root, train=False, transform=transform, download=download)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_length = len(train_dataset)
    val_length = len(val_dataset)
    print(f"\033[0;32;40mtrain images: {train_length}, val images: {val_length}\033[0m")

    return train_dataset, val_dataset, train_dataloader, val_dataloader


def cifar(
    variance="CIFAR10",
    batch_size=16,
    root="../datasets",
    download=True,
    resize=32,
    num_workers=8,
):
    """CIFAR10, CIFAR100

    Args:
        variance (str, optional):   数据集名称, [CIFAR10, CIFAR100]. Defaults to 'DTD'.
        batch_size (int, optional): batch_size. Defaults to 16.
        root (str, optional):       存放路径. Defaults to './dataset'.
        download (bool, optional):  是否下载数据集. Defaults to True.
        resize (int, optional):     图片大小. Defaults to 32.
        num_workers (int, optional):并行数. Defaults to 8.

    Returns:
        train_dataset, val_dataset, train_dataloader, val_dataloader
    """
    assert variance in [
        "CIFAR10",
        "CIFAR100",
    ], "variance should in CIFAR10 and CIFAR100"
    dataset = {
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
    }[variance]

    transform = v2.Compose(
        [
            v2.Resize(resize),
            transforms.ToTensor(),
            # v2.ToDtype(torch.float32, scale=True),  # 不会将PIL转换为Tensor
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = dataset(root, train=True, transform=transform, download=download)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_dataset = dataset(root, train=False, transform=transform, download=download)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_length = len(train_dataset)
    val_length = len(val_dataset)
    print(f"\033[0;32;40mtrain images: {train_length}, val images: {val_length}\033[0m")

    return train_dataset, val_dataset, train_dataloader, val_dataloader


def get_transforms(mode="Default", resize=224, valid_resize=224):
    """获取图片处理流程

    Args:
        mode (str, optional):         标准化类型, in [Default, Inception, DPN, MobileVit, OPENAI_CLIP]. Defaults to 'Default'.
        resize (int, optional):       训练图片大小. Defaults to 224.
        valid_resize (int, optional): 验证图片大小. Defaults to 224.

    Returns:
        [Transform]: 训练和测试处理流程
    """
    assert mode in ["Default", "Inception", "DPN", "MobileVit", "OPENAI_CLIP"]
    print(f"\033[0;36;40mtransform mode is {mode}!\033[0m")

    # refer: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/constants.py
    if mode == "Default":
        mean = [0.485, 0.456, 0.406]  # 减去均值(前面)
        std = [0.229, 0.224, 0.225]  # 除以标准差(后面)
    elif mode == "Inception":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif mode == "DPN":
        mean = [124 / 255, 117 / 255, 104 / 255]
        std = [1 / (0.0167 * 255)] * 3
    elif mode == "MobileVit":
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    elif mode == "OPENAI_CLIP":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    train_transform = v2.Compose(
        [
            # v2.Resize(int(resize/0.875)),         # 224 / 0.875 = 256
            # v2.RandomCrop((resize, resize)),
            v2.RandomResizedCrop(resize),  # 随机缩小剪裁,输出为(resize, resize)
            v2.RandomRotation(degrees=90),  # 随机旋转，-90到90度之间随机选
            v2.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
            v2.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            # v2.RandomPerspective(p=0.5),          # 透视变换
            # v2.RandomAffine(degrees=(-90, 90), translate=(0, 0.5), scale=(0.5, 1.5), shear=(-45, 45)),   # 随机仿射变化
            v2.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),  # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            # v2.RandomGrayscale(0.025),            # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),
            # v2.ToDtype(torch.float32, scale=True),  # 不会将PIL转换为Tensor
            v2.Normalize(mean, std),  # 归一化
        ]
    )

    valid_transform = v2.Compose(
        [
            v2.Resize(int(valid_resize / 0.875)),
            # v2.RandomCrop((valid_resize, valid_resize)),
            v2.CenterCrop((valid_resize, valid_resize)),
            transforms.ToTensor(),
            # v2.ToDtype(torch.float32, scale=True),  # 不会将PIL转换为Tensor
            v2.Normalize(mean, std),
        ]
    )

    return [train_transform, valid_transform]


def get_torchvision_cls_dataset(
    variance="DTD",
    batch_size=16,
    root="../datasets",
    download=True,
    resize=224,
    valid_resize=None,
    num_workers=8,
    transform_mode="Default",
) -> tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    """get torchvision classification dataset

    Args:
        variance (str, optional):   数据集名称,
            in [DTD, FGVCAircraft, Flowers102, Food101, ImageNet, Places365, StanfordCars, SUN397]. Defaults to 'DTD'.
        batch_size (int, optional):     batch_size. Defaults to 16.
        root (str, optional):           存放路径. Defaults to './dataset'.
        download (bool, optional):      是否下载数据集. Defaults to True.
        resize (int, optional):         训练图片大小. Defaults to 224.
        valid_resize (int, optional):   验证图片大小. Defaults to None.
        num_workers (int, optional):    并行数. Defaults to 8.
        transform_mode (str, optional): 标准化类型, in [Default, Inception, DPN, MobileVit, OPENAI_CLIP]. Defaults to 'Default'.

    Returns:
        train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader
    """
    assert variance in [
        "DTD",
        "FGVCAircraft",
        "Flowers102",
        "Food101",
        "ImageNet",
        "Places365",
        "StanfordCars",
        "SUN397",
    ], "variance should in DTD, FGVCAircraft, Flowers102, StanfordCars and ImageNet!"

    if valid_resize is None:
        valid_resize = resize
    train_transform, val_transform = get_transforms(
        transform_mode, resize, valid_resize
    )

    dataset = {
        "DTD": datasets.DTD,  # 5,640 images of 47 classes        train: 1880, val: 1880, test: 1880  596MB
        "FGVCAircraft": datasets.FGVCAircraft,  # 10,000 images of 100 classes      train: 3334, val: 3333, test: 3333  2.56GB
        "Flowers102": datasets.Flowers102,  # 8,189 images of 102 classes       train: 1020, val: 1020, test: 6149  328MB
        "Food101": datasets.Food101,  # 101,000 images of 101 classes
        "ImageNet": datasets.ImageNet,  # 1,200,000 images of 1000 classes
        "Places365": datasets.Places365,  # 10,000,000 images of 365 classes
        "StanfordCars": datasets.StanfordCars,  # 16,185 images of 196 classes
        "SUN397": datasets.SUN397,  # 130,519 images of 899 classes
    }[variance]

    train_dataset = dataset(
        root, split="train", transform=train_transform, download=download
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_dataset = dataset(root, split="val", transform=val_transform, download=download)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = dataset(
        root, split="test", transform=val_transform, download=download
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_length = len(train_dataset)
    val_length = len(val_dataset)
    test_length = len(test_dataset)
    print(
        f"\033[0;32;40mtrain images: {train_length}, val images: {val_length}, test images: {test_length}\033[0m"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )


# -------------------#
#   显示1通道的图片
# -------------------#
def show_1_channel(dataloader: DataLoader):
    import matplotlib.pyplot as plt
    import numpy as np

    # 获取一批数据
    image_target = next(iter(dataloader))
    print(len(image_target))  # 2             images+targets
    print(len(image_target[0]))  # 1024          images

    # 均值，标准差
    mean, std = 0.485, 0.229

    # 24张图片
    target_list = image_target[1][:24]  # target
    target_list = [target.numpy() for target in target_list]  # target tensor to numpy
    image_list = image_target[0][:24]  # image
    img_list = []
    for image in image_list:
        image = image.numpy()
        np.transpose(image, (1, 2, 0))  # c,h,w -> h,w,c
        image = image * std + mean  # 反标准化  * 标准差 + 均值
        image = (image * 255).astype(np.uint8)
        img_list.append(image)

    print(img_list[0].shape)  # (32, 32, 3)

    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(18, 9), dpi=100)
    for i in range(24):
        row = i // 8  # 0 1 2
        col = i % 8  # 0 1 2 3 4 5 6 7
        axes[row][col].set_title(target_list[i])
        axes[row][col].imshow(img_list[i])
        # axes[row][col].imshow(img_list[i], cmap='gray')   # 灰度图像
        # axes[row][col].set_xlim(0, 31)
        # axes[row][col].set_ylim(31, 0)    31 0 不然图片上下颠倒
    plt.show()


# -------------------#
#   显示3通道的图片
# -------------------#
def show_3_channels(dataLoader: DataLoader):
    import matplotlib.pyplot as plt
    import numpy as np

    # 获取一批数据
    image_target = next(iter(dataLoader))
    print(len(image_target))  # 2             images+targets
    print(len(image_target[0]))  # 1024          images

    # 单张图片画图
    # image = image_target[0][0]
    # print(image.size())               # [3, 32, 32]   a image
    # 转置
    # image = image.permute(1, 2, 0)
    # plt.imshow(image)
    # plt.show()

    # 均值，标准差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 24张图片
    target_list = image_target[1][:24]  # target
    target_list = [target.numpy() for target in target_list]  # target tensor to numpy
    image_list = image_target[0][:24]  # image
    img_list = []
    for image in image_list:
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))  # c,h,w -> h,w,c
        image = image * std + mean  # 反标准化,每个通道单独处理  * 标准差 + 均值
        image = (image * 255).astype(np.uint8)
        img_list.append(image)

    print(img_list[0].shape)  # (32, 32, 3)

    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(18, 9), dpi=100)
    for i in range(24):
        row = i // 8  # 0 1 2
        col = i % 8  # 0 1 2 3 4 5 6 7
        axes[row][col].set_title(target_list[i])
        axes[row][col].imshow(img_list[i])
        # axes[row][col].imshow(img_list[i], cmap='gray')   # 灰度图像
        # axes[row][col].set_xlim(0, 31)
        # axes[row][col].set_ylim(31, 0)    31 0 不然图片上下颠倒
    plt.show()


if __name__ == "__main__":
    _, _, train_dataloader, _ = cifar(
        "CIFAR10", root="../datasets", batch_size=24, resize=32
    )
    image_target = next(iter(train_dataloader))
    print(image_target[0].shape)  # [24, 3, 32, 32]
    print(
        image_target[1]
    )  # [7, 2, 6, 8, 9, 9, 6, 0, 3, 4, 2, 1, 1, 3, 6, 8, 2, 2, 8, 3, 0, 1, 7, 2]

    show_3_channels(train_dataloader)

    # _, _, _, train_dataloader, _, _ = get_torchvision_cls_dataset()
    # image_target = next(iter(train_dataloader))
    # print(image_target[1])    # [39, 42,  1, 10, 31, 34, 15, 11, 38, 37,  1, 37, 27, 21, 18,  3]
