import time
import torch
from torch import nn, optim
from modified_model.mobilenetv3 import mobilenet_v3_large
import os
from utils.train_lightning import lightning_trainer
from utils.dataset_pre import cifar
from utils.utils import save_file


EPOCHS      = 100
LR          = 0.001
RESIZE      = 32
BATCH_SIZE  = 128
NUM_CLASSES = 10
PRETRAINED  = True
SAVE_PATH   = f"./checkpoints/{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-mobilenet_v3_large_d8"
ACCELERATOR = "gpu"     # ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
DEVICES     = 1         # gpu数量
PRECISION   = "32-true" # ("16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true")
SYNC_BN     = False


if __name__ == "__main__":
    # 保存当前文件
    file_name = os.path.basename(__file__)
    save_file(file_name, SAVE_PATH)

    # dataset
    _, _, train_dataloader, val_dataloader = cifar(variance="CIFAR10", batch_size=BATCH_SIZE, resize=RESIZE, root="./datasets")

    # model
    model = mobilenet_v3_large(pretrained=PRETRAINED)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.937, weight_decay=5e-4)
    # 学习率衰减 使用step迭代
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dataloader), eta_min=LR*0.01)    # 最小减小100倍

    lightning_trainer(
        epochs              = EPOCHS,
        model               = model,
        optimizer           = optimizer,
        lr_scheduler        = lr_scheduler,
        lr_scheduler_tactic = "step",   # 学习率衰减 使用step迭代
        train_dataloader    = train_dataloader,
        val_dataloader      = val_dataloader,
        save_path           = SAVE_PATH,
        accelerator         = ACCELERATOR,
        devices             = DEVICES,
        precision           = PRECISION,
        sync_batchnorm      = SYNC_BN,
        num_classes         = NUM_CLASSES,
    )
