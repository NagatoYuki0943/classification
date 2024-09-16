import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torchmetrics.functional import accuracy, precision, recall, confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    DeviceStatsMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
# import lightning as pl
# from lightning import Trainer
# from lightning.pytorch.callbacks import LearningRateMonitor, DeviceStatsMonitor, ModelCheckpoint, EarlyStopping
# from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger


# pl.seed_everything(42)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_tactic: str = "epoch",
        num_classes: int | None = None,
    ) -> None:
        """
        Args:
            model (Module): 模型
            optimizer (Optimizer, optional): 优化器. Defaults to None.
            lr_scheduler (LRScheduler, optional): 学习率调整. Defaults to None.
            lr_scheduler_tactic (str, optional): 学习率调整策略, epoch or step. Defaults to epoch.
            num_classes (int, optional): 分类数. Defaults to None.
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_tactic = lr_scheduler_tactic
        self.num_classes = num_classes

        # 保存step的输出
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        """返回优化器和学习率调度器

        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers
        """
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": self.lr_scheduler_tactic,
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
            },
        }

    # 自定义函数
    def step(self, batch):
        """训练和验证的一个step
        Args:
            batch: images, target
        return:
            loss, preds, target
        """
        images, targets = batch
        preds: Tensor = self(images)
        loss = F.cross_entropy(preds, targets)
        # 获取预测id,用来计算,precision,recall
        _, preds = preds.max(dim=-1)
        return loss, preds, targets

    def epoch_end(self, step_outputs):
        """训练和验证一个epoch结尾处数据处理
        Args:
            step_outputs: step_outputs返回的数据列表  [(loss, preds, targets),...]
        return:
            loss_avg, acc_avg, precision, recall, confusion_matrix
        """
        # stack可以拼接0维数据,cat不行
        loss_avg = torch.stack([x[0] for x in step_outputs]).mean()
        preds = torch.cat([x[1] for x in step_outputs], dim=0)
        targets = torch.cat([x[2] for x in step_outputs], dim=0)
        acc_avg = accuracy(
            preds=preds,
            target=targets,
            task="multiclass",
            num_classes=self.num_classes,
            top_k=1,
        )
        p = precision(
            preds=preds,
            target=targets,
            task="multiclass",
            num_classes=self.num_classes,
            average=None,
        )
        r = recall(
            preds=preds,
            target=targets,
            task="multiclass",
            num_classes=self.num_classes,
            average=None,
        )
        cm = confusion_matrix(
            preds=preds,
            target=targets,
            task="multiclass",
            num_classes=self.num_classes,
            normalize="true",
        )
        return loss_avg, acc_avg, p, r, cm

    # -----------------------------------------------------------------#

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        """Called at the beginning of training after sanity check."""
        pass

    def on_train_epoch_start(self):
        """在一个训练epoch开始处被调用"""
        pass

    def training_step(self, batch, *args, **kwargs):
        loss, preds, targets = self.step(batch)
        self.train_step_outputs.append((loss, preds, targets))
        acc = accuracy(
            preds=preds,
            target=targets,
            task="multiclass",
            num_classes=self.num_classes,
            top_k=1,
        )
        self.log("train/acc_iter", acc, prog_bar=True)
        self.log("train/loss_iter", loss, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        """在一个训练epoch结尾处被调用"""
        loss_avg, acc_avg, _, _, _ = self.epoch_end(self.train_step_outputs)
        self.log("train/loss", loss_avg)
        self.log("train/acc", acc_avg)
        # free up the memory
        self.train_step_outputs.clear()

    # -----------------------------------------------------------------#

    def validation_step(self, batch, *args, **kwargs):
        loss, preds, targets = self.step(batch)
        self.val_step_outputs.append((loss, preds, targets))
        acc = accuracy(
            preds=preds,
            target=targets,
            task="multiclass",
            num_classes=self.num_classes,
            top_k=1,
        )
        self.log("val/acc_iter", acc, prog_bar=True)
        self.log("val/loss_iter", loss, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        """在一个验证epoch结尾处被调用"""
        loss_avg, acc_avg, _, _, _ = self.epoch_end(self.val_step_outputs)
        self.log("val/loss", loss_avg)
        self.log("val/acc", acc_avg)
        # free up the memory
        self.val_step_outputs.clear()

    # -----------------------------------------------------------------#

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        pass

    def test_step(self, batch, *args, **kwargs):
        loss, preds, targets = self.step(batch)
        self.test_step_outputs.append((loss, preds, targets))
        acc = accuracy(
            preds=preds,
            target=targets,
            task="multiclass",
            num_classes=self.num_classes,
            top_k=1,
        )
        self.log("test/acc_iter", acc, prog_bar=True)
        self.log("test/loss_iter", loss, prog_bar=True)
        return {"loss": loss}

    def on_test_epoch_end(self):
        """在一个测试epoch结尾处被调用"""
        loss_avg, acc_avg, precision, recall, confusion_matrix = self.epoch_end(
            self.test_step_outputs
        )
        self.log("test/loss", loss_avg)
        self.log("test/acc", acc_avg)
        # free up the memory
        self.test_step_outputs.clear()

        # 打印全部数据
        np.set_printoptions(precision=4, threshold=np.inf)
        torch.set_printoptions(precision=4, profile="full")
        print("=" * 100)
        print(f"acc: {acc_avg.cpu().detach().numpy()}")
        print("-" * 100)
        print(f"precision: {precision.cpu().detach().numpy()}")
        print("-" * 100)
        print(f"recall:  {recall.cpu().detach().numpy()}")
        print("-" * 100)
        print(f"confusion_matrix:\n", confusion_matrix.cpu().detach().numpy())
        print("=" * 100)


def lightning_trainer(
    epochs: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: optim.lr_scheduler.LRScheduler,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader = None,
    lr_scheduler_tactic: str = "epoch",
    save_path: str = "checkpoints",
    accelerator: str = "gpu",
    devices: int = 1,
    precision: int | str = "32-true",
    sync_batchnorm: bool = False,
    num_classes: int = 0,
    project: str | None = None,
):
    """封装的lightning训练函数

    Args:
        epochs (int):                           训练的epoch数
        model (nn.Module):                      模型
        optimizer (Optimizer):                  优化器
        lr_scheduler (LRScheduler):             学习率调整策略
        train_dataloader (DataLoader):          训练数据集
        val_dataloader (DataLoader):            验证数据集
        test_dataloader (DataLoader, optional): 测试数据集. Defaults to None.
        lr_scheduler_tactic (str, optional):    学习率调整策略更新方式, epoch or step. Defaults to epoch.
        save_path (str, optional):              保存路径. Defaults to checkpoints.
        accelerator (string, optional):         加速器. ("cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto"). Defaults to gpu.
        devices (int, optional):                gpu数量, None为CPU. Defaults to 1.
        precision (int | str, optional):        训练精度. ("transformer-engine", "transformer-engine-float16", "16-true", "16-mixed",
                                                "bf16-true", "bf16-mixed", "32-true", "64-true") Defaults to "32-true".
        sync_batchnorm (bool, optional):        同步批量归一化. Defaults to False.
        num_classes(int, optional):             分类数. Defaults to 0.
        project (str | None, optional):         项目名称. Defaults to None.
    """
    assert (
        num_classes > 0
    ), "\033[0;31;40mnum_classes must be an integer greater than 0 !\033[0m"

    print(
        f"\033[0;34;40m==================== training files save to {save_path} ====================\033[0m"
    )

    lmodel = LightningTrainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_tactic=lr_scheduler_tactic,
        num_classes=num_classes,
    )

    # 保存最好的模型
    model_checkpoint = ModelCheckpoint(
        dirpath=f"{save_path}/checkpoints",
        monitor="val/acc",
        mode="max",  # 使用准确率作为判断标准要将mode设置为max,保存准确率最高的模型
        save_top_k=1,
        filename="best_acc-{epoch:03d}",
        save_last=True,
    )

    # 早停
    early_stopping = EarlyStopping(
        monitor="val/acc",
        patience=100,  # 100轮没提升就停止
        mode="max",  # 不再增长就停止
    )

    # 回调
    callbacks = [
        LearningRateMonitor(logging_interval="step", log_momentum=True),
        DeviceStatsMonitor(cpu_stats=True),
        model_checkpoint,
        early_stopping,
    ]

    hyperparams = {
        "model": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "lr_scheduler": lr_scheduler.__class__.__name__,
        "init_lr": optimizer.state_dict()["param_groups"][0]["lr"],
        "batch_size": train_dataloader.batch_size,
        "train_size": list(next(iter(train_dataloader))[0].size()[2:]),
        "val_size": list(next(iter(val_dataloader))[0].size()[2:]),
    }

    # loggder
    loggder = [
        CSVLogger(save_dir=save_path, name="csv_logs", version="")
    ]  # name="" 保存路径就为 save_dir/version
    tensorboard_logger = TensorBoardLogger(
        save_dir=save_path, name="tensorboard_logs", version=""
    )
    tensorboard_logger.log_hyperparams(hyperparams)
    loggder.append(tensorboard_logger)
    print(
        f"\033[0;32;40m====== tensorboard log success, please use `tensorboard --logdir=tensorboard_logs` in {save_path} to check metrics ======\033[0m"
    )

    # wandb https://wandb.ai/site
    try:
        wandb_loggder = WandbLogger(
            save_dir=save_path,
            project=project
            if project is not None
            else train_dataloader.dataset.__class__.__name__,
        )
        wandb_loggder.log_hyperparams(hyperparams)
        loggder.append(wandb_loggder)
    except:
        print(
            "\033[0;33;40m================== wandb log fail, please use `pip install wandb` to install wandb(optional) ==================\033[0m"
        )

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=loggder,
        callbacks=callbacks,
        max_epochs=epochs,
        # val_check_interval = 1.0,  # 验证比例 0.25表示每隔四分之一的epoch做一次validation，
        #                            # 1000表示每隔1000个steps做一次validation(1表示每个step做一次val,所以要用1.0)
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        sync_batchnorm=sync_batchnorm,
        default_root_dir=save_path,
    )

    # train&val
    trainer.fit(lmodel, train_dataloader, val_dataloader)

    # test
    if test_dataloader is None:
        test_dataloader = val_dataloader
    trainer.test(lmodel, test_dataloader)


def load_lightning_model(model: nn.Module, path: str) -> pl.LightningModule:
    """获取训练权重

    Args:
        model (Module): 模型,因为LightningTrainer需要这个参数初始化，所以必须给,optimizer同理，不过用不到就给了None
        path (str):     保存的权重路径
    Returns:
        pl.LightningModule:  读取的模型
    """
    lightning_model = LightningTrainer.load_from_checkpoint(path, model=model)
    return lightning_model
