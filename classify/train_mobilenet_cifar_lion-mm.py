import os

import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

from utils.dataset_pre import cifar
from utils.utils import save_file

from mmengine.runner import Runner
from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, METRICS


EPOCHS      = 100
LR          = 0.0001 # Lion default lr
RESIZE      = 32
BATCH_SIZE  = 128
NUM_CLASSES = 10
PRETRAINED  = True
SAVE_PATH   = f'./checkpoints/mobilenet_v3_large-cifar-lion-mm'
PRECISION   = 16 # (16, 32)


# https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/model.md
@MODELS.register_module(name='MobileNet')
class MobileNet(BaseModel):
    def __init__(self, variant: str='mobilenet_v3_large', num_classes: int=1000):
        super().__init__()
        assert variant in ['mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']

        self.model: nn.Module
        if variant == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        elif variant == 'mobilenet_v3_large':
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        elif variant == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # change num_classes
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    # custom forward
    def forward(self, imgs, labels, mode):
        x = self.model(imgs)
        if mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}
        elif mode == 'predict':
            return x, labels


# https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html
@METRICS.register_module('Accuracy')
class Accuracy(BaseMetric):
    def process(self, data_batch, data_samples):
        score, gt = data_samples
        # 将一个批次的中间结果保存至 `self.results`
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(item['correct'] for item in results)
        total_size = sum(item['batch_size'] for item in results)
        # 返回保存有评测指标结果的字典，其中键为指标名称
        return dict(accuracy=100 * total_correct / total_size)


if __name__ == '__main__':
    # 保存当前文件
    file_name = os.path.basename(__file__)
    save_file(file_name, SAVE_PATH)

    # dataset
    train_dataset, val_dataset, _, _ = cifar(variance='CIFAR10', batch_size=BATCH_SIZE, resize=RESIZE, root='./datasets')

    # https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html
    runner = Runner(
        # custom模型
        # model = MobileNet(variant='mobilenet_v3_large', num_classes=NUM_CLASSES),
        model = dict(type='MobileNet', variant='mobilenet_v3_large', num_classes=NUM_CLASSES),

        # 模型检查点、日志等都将存储在工作路径中
        work_dir = SAVE_PATH,

        # 优化器封装，MMEngine 中的新概念，提供更丰富的优化选择。
        # 通常使用默认即可，可缺省。有特殊需求可查阅文档更换，如
        # "AmpOptimWrapper" 开启混合精度训练
        optim_wrapper = dict(
            type = 'AmpOptimWrapper' if PRECISION == 16 else 'OptimWrapper',
            # 如果你想要使用 BF16，请取消下面一行的代码注释
            # dtype='bfloat16',  # 可用值： ('float16', 'bfloat16', 'float32', 'float64', None)
            optimizer=dict(
                type='Lion',
                lr=LR,
                betas=(0.9, 0.99),
                weight_decay=0.0,
            ),
            # accumulative_counts=2,          # 梯度累加
            # clip_grad=dict(max_norm=1),     # 基于 torch.nn.utils.clip_grad_norm_  对梯度进行裁减
            # clip_grad=dict(clip_value=0.2), # 基于 torch.nn.utils.clip_grad_value_ 对梯度进行裁减
        ),

        # 参数调度器，用于在训练中调整学习率/动量等参数
        # https://mmengine.readthedocs.io/zh_CN/latest/tutorials/param_scheduler.html
        param_scheduler = [
            # warmup
            dict(
                type='LinearLR',
                start_factor=0.001,
                by_epoch=True,
                begin=0,
                end=5,
                convert_to_iter_based=True, # 转换为基于iter
            ),
            # cos
            dict(
                type='CosineAnnealingLR',
                T_max=EPOCHS-5,             # 需要调整T_max
                eta_min=LR*0.01,
                by_epoch=True,
                begin=5,
                end=EPOCHS,
                convert_to_iter_based=True, # 转换为基于iter
            )
        ],

        # 训练所用数据
        # https://mmengine.readthedocs.io/zh_CN/latest/tutorials/dataset.html#dataset-dataloader
        # https://mmengine.readthedocs.io/zh_CN/latest/tutorials/dataset.html#sampler-shuffle
        train_dataloader = dict(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=8,
            collate_fn=dict(type='default_collate'),
            # shuffle=True,
            sampler=dict(     # DataLoader中如果设置sampler,shuffle就不要设置
                type='DefaultSampler',
                shuffle=True,
            ),
        ),

        # 训练相关配置
        train_cfg = dict(
            type='EpochBasedTrainLoop',
            max_epochs=EPOCHS,
            val_begin=0,     # 从第 0 个 epoch 开始验证
            val_interval=1,  # 每隔 1 个 epoch 进行一次验证
        ),

        # 验证所用数据
        val_dataloader = dict(
            dataset=val_dataset,
            batch_size=1,
            num_workers=2,
            collate_fn=dict(type='default_collate'),
            # shuffle=False,
            sampler=dict(     # DataLoader中如果设置sampler,shuffle就不要设置
                type='DefaultSampler',
                shuffle=False,
            ),
        ),

        # 验证相关配置，通常为空即可
        val_cfg = dict(),

        # 验证指标与验证器封装，可自由实现与配置
        val_evaluator = dict(type='Accuracy'),

        # 钩子属于进阶用法，如无特殊需要，尽量缺省
        default_hooks = dict(
            runtime_info=dict(type='RuntimeInfoHook'),
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook', interval=100),
            # https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html#checkpointhook
            checkpoint=dict(
                type='CheckpointHook',
                interval=1,
                max_keep_ckpts=5,
                by_epoch=True,
                save_last=True,
                save_best='auto',
                # published_keys=['meta', 'state_dict'],
            ),
        ),

        # runtime setting
        # custom_hooks = [
        #     dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL'),
        # ],

        visualizer = dict(
            type='Visualizer',
            vis_backends=[
                dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend'),
                # dict(type='WandbVisBackend'),     # 使用 WandB 前需安装依赖库 wandb 并登录至 wandb, wandb login
            ],
        ),

        log_level = 'INFO',
        log_processor = dict(by_epoch=True),

        # # `luancher` 与 `env_cfg` 共同构成分布式训练环境配置
        # launcher='pytorch',

        env_cfg=dict(
            cudnn_benchmark=False,                        # 是否使用 cudnn_benchmark
            dist_cfg=dict(backend='nccl', timeout=1800),  # 分布式通信后端
            mp_cfg=dict(
                mp_start_method='fork',
                opencv_num_threads=0
            )
        ),

        # 加载权重的路径 (None 表示不加载)
        load_from = None,

        # 从加载的权重文件中恢复训练
        # 设置 Runner 的 resume 等于 True，Runner 会从 work_dir 中加载最新的 checkpoint
        # 如果希望指定恢复训练的路径，除了设置 resume=True，还需要设置 load_from 参数。
        # 需要注意的是，如果只设置了 load_from 而没有设置 resume=True，则只会加载
        # checkpoint 中的权重并重新开始训练，而不是接着之前的状态继续训练
        # https://mmengine.readthedocs.io/zh_CN/latest/common_usage/resume_training.html
        resume = False,

        # randomness 设置
        randomness = dict(seed=43),
    )

    # 开始训练你的模型吧
    runner.train()
