import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from .dataset_pre import get_transforms
import json


def get_custom_data(
    dir: str,
    batch_size=8,
    resize=224,
    valid_resize=None,
    num_workers=8,
    transform_mode="Default",
) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """根据路径获取数据集

    Args:
        data_dir (str): 数据集目录
        batch_size (int, optional):     batch. Defaults to 8.
        resize (int, optional):         训练图片大小. Defaults to 224.
        valid_resize (int, optional):   验证图片大小. Defaults to None.
        num_workers (int, optional):    图片读取线程数. Defaults to 8.
        transform_mode (str, optional): 标准化类型, in [Default, Inception, DPN, MobileVit, OPENAI_CLIP]. Defaults to 'Default'.

    Returns:
        train_dataset, val_dataset, train_dataloader, val_dataloader
    """
    if valid_resize is None:
        valid_resize = resize
    train_transform, val_transform = get_transforms(
        transform_mode, resize, valid_resize
    )

    train_dataset = ImageFolder(
        root=os.path.join(dir, "train"), transform=train_transform
    )
    val_dataset = ImageFolder(root=os.path.join(dir, "val"), transform=val_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    train_length = len(train_dataset)
    val_length = len(val_dataset)
    print(f"\033[0;32;40mtrain images: {train_length}, val images: {val_length}\033[0m")

    # 将id和标签写入json列表
    with open(f"{dir}/class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset.class_to_idx, f, ensure_ascii=False, indent=4)
        print(f"\033[0;35;40m{dir}/class_to_idx.json write success.\033[0m")

    return train_dataset, val_dataset, train_dataloader, val_dataloader


# ---------------------------------------------#
#   显示3通道的图片
# ---------------------------------------------#
def show_3_channels():
    train_dataset, val_dataset, train_dataloader, val_dataloader = get_custom_data(
        "./datasets/scenery", 24, 128
    )

    # 获取文件夹名字和标签字典
    class_to_idx = train_dataset.class_to_idx
    print(
        class_to_idx
    )  # {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

    # 获取文件夹列表
    classes = train_dataset.classes
    print(classes)  # ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

    # 通过id找到文件夹
    print(classes[0])  # buildings

    # 读取json并切换key和value的位置
    with open("./datasets/scenery/class_to_idx.json", "r", encoding="utf-8") as f:
        cls_to_idx = json.load(f)
    idx_to_cls = dict(zip(cls_to_idx.values(), cls_to_idx.keys()))
    print(
        idx_to_cls
    )  # {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
    print("*" * 100)

    # -------------------------------#
    #   画图
    # -------------------------------#
    import matplotlib.pyplot as plt
    import numpy as np

    # 获取一批数据
    image_target = next(iter(train_dataloader))
    print(len(image_target))  # 2             images+targets
    print(len(image_target[0]))  # 12            images

    # 均值，标准差
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # -------------------------------#
    #   24张图片
    # -------------------------------#
    target_list = image_target[1]
    target_list = [target.numpy() for target in target_list]  # target
    target_list = [
        idx_to_cls[int(target)] for target in target_list
    ]  # target id to name
    image_list = image_target[0]  # image
    img_list = []
    for image in image_list:
        image = image.numpy()
        # 反标准化，每个通道单独处理， *标准差 + 均值
        for i in range(3):
            image[i] = image[i] * std[i] + mean[i]
        # c,h,w -> h,w,c
        img_list.append(np.transpose(image, (1, 2, 0)))

    print(img_list[0].shape)  # (32, 32, 3)

    fig, axes = plt.subplots(nrows=3, ncols=8, figsize=(18, 9), dpi=100)
    for i in range(24):
        row = i // 8  # 0 1 2
        col = i % 8  # 0 1 2 3 4 5 6 7
        axes[row][col].set_title(target_list[i])
        axes[row][col].imshow(img_list[i])
        # axes[row][col].set_xlim(0, 31)
        # axes[row][col].set_ylim(31, 0)    31 0 不然图片上下颠倒
    plt.show()


if __name__ == "__main__":
    show_3_channels()
