import time
import torch
from torch import nn, optim
from torchvision import models
import os
from utils.train_lightning import lightning_trainer
from utils.dataset_pre import get_torchvision_cls_dataset
from utils.utils import save_file


EPOCHS = 100
LR = 0.001
RESIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 47
PRETRAINED = True
SAVE_PATH = f"./checkpoints/{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-resnet50"
ACCELERATOR = "gpu"  # ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
DEVICES = 1  # gpu数量
PRECISION = "32-true"  # ("16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true")
SYNC_BN = False


if __name__ == "__main__":
    # 保存当前文件
    file_name = os.path.basename(__file__)
    save_file(file_name, SAVE_PATH)

    # dataset
    _, _, _, train_dataloader, val_dataloader, test_dataloader = (
        get_torchvision_cls_dataset("DTD", BATCH_SIZE, resize=RESIZE, root="./datasets")
    )

    # model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.937, weight_decay=5e-4)
    # 学习率衰减 使用step迭代
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_dataloader), eta_min=LR * 0.01
    )  # 最小减小100倍

    lightning_trainer(
        epochs=EPOCHS,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_tactic="step",  # 学习率衰减 使用step迭代
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        save_path=SAVE_PATH,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        sync_batchnorm=SYNC_BN,
        num_classes=NUM_CLASSES,
    )
