import time
import torch
from torch import nn, optim
from timm import models
import os
from utils.train_lightning import lightning_trainer
from utils.dataset import get_custom_data
from utils.utils import save_file


EPOCHS      = 100
LR          = 0.001
RESIZE      = 256
BATCH_SIZE  = 64
NUM_CLASSES = 6
PRETRAINED  = True
SAVE_PATH   = f"./checkpoints/{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-mobilevit_s"
ACCELERATOR = "gpu"     # ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto")
DEVICES     = 1         # gpu数量
PRECISION   = "32-true" # ("16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true")
SYNC_BN     = False


if __name__ == "__main__":
    # 保存当前文件
    file_name = os.path.basename(__file__)
    save_file(file_name, SAVE_PATH)

    # dataset
    _, _, train_dataloader, val_dataloader = get_custom_data("datasets/intel_origin_scenery", batch_size=BATCH_SIZE, resize=RESIZE, transform_mode="MobileVit")

    # model
    model = models.mobilevit.mobilevit_s(pretrained=True, num_classes=NUM_CLASSES)

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
