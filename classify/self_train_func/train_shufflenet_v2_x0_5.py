import time
import torch
from torch import nn, optim
from torchvision import models
from utils.train import train_epochs
from utils.dataset import get_data
from utils.dataset_pre import cifar


EPOCHS      = 100
LR          = 0.001
BATCH_SIZE  = 128
NUM_CLASSES = 100
PRETRAINED  = True
SAVE_PATH   = f"./checkpoints/{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-shufflenet_v2_x0_5"
CUDA        = True
HALF        = False
DISTRIBUTED = False
SYNC_BN     = False


if __name__ == "__main__":
    # dataset
    # datasets, _, _ = get_data("./dataset/scenery", batch_size=BATCH_SIZE, resize=128)
    # train_dataset, val_dataset = datasets['train'], datasets['val']
    # dataset
    train_dataset, val_dataset, _, _ = cifar(variance='CIFAR100', batch_size=BATCH_SIZE)

    # model
    model       = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
    model.fc    = nn.Linear(model.fc.in_features, NUM_CLASSES)

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