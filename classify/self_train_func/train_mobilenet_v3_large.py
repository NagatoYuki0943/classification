import time
import torch
from torch import nn, optim
from torchvision import models
from utils.train import train_epochs
from utils.dataset_pre import cifar


EPOCHS      = 100
LR          = 0.001
BATCH_SIZE  = 128
NUM_CLASSES = 10
PRETRAINED  = True
SAVE_PATH   = f"./checkpoints/{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-mobilenet_v3_large-resize64"
CUDA        = True
HALF        = False
DISTRIBUTED = True
SYNC_BN     = False


if __name__ == "__main__":
    # dataset
    train_dataset, val_dataset, _, _ = cifar(batch_size=BATCH_SIZE, resize=64, num_workers=4)

    # model
    model       = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)

    # 优化器
    optimizer   = optim.AdamW(model.parameters(), lr=LR)
    # optimizer   = optim.SGD(model.parameters(), lr=LR, momentum=0.937, weight_decay=5e-4)
    # 学习率衰减
    lr_sche     = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.01)    # 最小减小100倍
    loss_fn     = nn.CrossEntropyLoss()

    #----------------------------------------#
    #   训练
    #----------------------------------------#
    train_epochs(
        epochs          = EPOCHS,
        model           = model,
        optimizer       = optimizer,
        loss_fn         = loss_fn,
        cuda            = CUDA,
        train_dataset   = train_dataset,
        val_dataset     = val_dataset,
        batch_size      = BATCH_SIZE,
        num_workers     = 0,
        save_path       = SAVE_PATH,
        save_interval   = 10,
        lr_sche         = lr_sche,
        half            = HALF,
        distributed     = DISTRIBUTED,
        sync_bn         = SYNC_BN,
    )