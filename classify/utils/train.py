import os
import time
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
from utils.utils import check_dir_exist, record, show_config
from torch.cuda.amp import GradScaler


def train_epochs(epochs: int, model: Module, optimizer: Optimizer, loss_fn: Module, cuda: bool,
                 train_dataset: Dataset, val_dataset: Dataset, batch_size: int, num_workers=0,
                 save_path="./checkpointss/", save_interval=10, lr_sche: LRScheduler=None,
                 half=False, distributed=False, sync_bn=False):
    """循环训练

    Args:
        epochs (int):                   训练总轮数
        model (Module):                 训练的模型
        optimizer (Optimizer):          优化器
        loss_fn (Module):               损失函数
        cuda (bool):                    是否使用GPU
        train_dataset (Dataset):        训练dataset
        val_dataset (Dataset):          验证dataset
        batch_size (int):               batch_size
        num_workers (int, optional):    读取图片线程数. Defaults to 0.
        save_path (str, optional):      模型保存路径. Defaults to "./checkpoints/".
        save_interval (int, optional):  保存模型间隔. Defaults to 10.
        lr_sche (LRScheduler, optional):学习率衰减. Defaults to None.
        half (bool, optional):          半精度训练. Defaults to False.
        distributed (bool, optional):   多显卡训练. Defaults to False.
            DP模式:
                设置            distributed = False
                在终端中输入     CUDA_VISIBLE_DEVICES=0,1 python train.py
            DDP模式:
                设置            distributed = True
                在终端中输入     CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use_env --nproc_per_node=2 train.py
                                CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use_env --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint='127.0.0.1:29400' train.py
                                CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
                                CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint='127.0.0.1:29400'  train.py
        sync_bn (bool, optional):       同步BN. Defaults to False.
    """

    #------------------------------------------------------#
    #   from https://github.com/bubbliiiing/classification-pytorch
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    elif cuda:
        print('Single GPU train')
        device          = torch.device('cuda:0')
        local_rank      = 0
        rank            = 0
    else:
        print('Single CPU train')
        device          = torch.device('cpu')
        local_rank      = 0
        rank            = 0
        half            = False     # 使用cpu时不支持half

    if local_rank == 0:
        # 创建需要的文件夹
        check_dir_exist([save_path])

        #----------------------------------------------------#
        #   显示当前config
        #----------------------------------------------------#
        # 优化器参数
        optim_param = [k+"="+str(v) for k, v in optimizer.state_dict()['param_groups'][0].items() if 'params' not in k]
        # 学习率降低参数
        if lr_sche:
            lr_param = [k+"="+str(v) for k, v in lr_sche.state_dict().items() if 'params' not in k]
        show_config(epochs=epochs, save_interval=save_interval, device=device, model=model.__class__.__name__,
                    optimizer=optimizer.__class__.__name__ + ": " + " ".join(optim_param),
                    lr_sche=lr_sche.__class__.__name__ + ": " + " ".join(lr_param) if lr_sche else None,
                    loss_fn=loss_fn.__class__.__name__, batch_size=batch_size, half=half,
                    distributed=distributed, sync_bn=sync_bn, save_path=save_path)

    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    #----------------------------#
    #   多卡平行运行
    #----------------------------#
    if distributed:
        model = model.cuda(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model.to(device)

    #----------------------------#
    #   创建dataset
    #----------------------------#
    if distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size      = batch_size # batch_size // ngpus_per_node 不将batch_size除以总显卡数，所以每张卡的数量都是指定的数量
        shuffle         = False
    else:
        train_sampler   = None
        val_sampler     = None
        shuffle         = True
    train_dataloader    = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=False, sampler=train_sampler)
    val_dataloader      = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=False, sampler=val_sampler)

    #----------------------------------------------------#
    #   半精度scaler
    #----------------------------------------------------#
    if cuda and half:
        scaler = GradScaler()

    #----------------------------------------------------#
    #   实例化SummaryWriter对象
    #   参数是保存的目录
    #----------------------------------------------------#
    tb_writer = SummaryWriter(log_dir=save_path+'/tf_logs')

    BEST_ACC = 0.0

    for epoch in range(1, epochs + 1):   # 1~epochs
        start = time.time()
        if local_rank == 0:
            # 创建进度条 长度                         描述                                       添加其余数据的方式
            pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{epochs} train')
        #----------------------------------------#
        #   train
        #----------------------------------------#
        model.train()
        train_loss_arr = []
        for image, target in train_dataloader:
            if distributed:
                image, target = image.cuda(local_rank), target.cuda(local_rank)
            else:
                image, target = image.to(device), target.to(device)

            # 反向传播
            optimizer.zero_grad()
            if not cuda and not half:
                #--------------------------#
                #   fp32
                #--------------------------#
                predict: Tensor = model(image)
                loss: Tensor    = loss_fn(predict, target)
                loss.backward()
                optimizer.step()
            else:
                #--------------------------#
                #   half
                #--------------------------#
                                                                # https://zhuanlan.zhihu.com/p/348554267
                with torch.autocast(device_type="cuda"):        # 在autocast enable区域运行forward
                    predict: Tensor = model(image)              # model做一个FP16的副本，forward
                    loss: Tensor    = loss_fn(predict, target)
                # loss.backward()
                # ptimizer.step()
                scaler.scale(loss).backward()                   # 用scaler，scale loss(FP16)，backward得到scaled的梯度(FP16)
                scaler.step(optimizer)                          # scaler 更新参数，会先自动unscale梯度，如果有nan或inf，自动跳过
                scaler.update()                                 # scaler factor更新

            train_loss_arr.append(loss.item())

            if local_rank == 0:
                pbar.update(1)
                # 在每一行后面添加数值
                pbar.set_postfix(**{'loss': loss.item()})
                # Epoch 3/100:  26%|███            | 3227/12500 [00:06<00:24, 379.15it/s, accuracy=2, lr=3, total_loss=1]
        # 学习率
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        # 学习率衰减
        if lr_sche is not None:
            lr_sche.step()

        if local_rank == 0:
            pbar.close()
            train_avg_loss = torch.as_tensor(train_loss_arr).mean().item()
            print(f'Epoch {epoch}/{epochs}', 'train_avg_loss:', train_avg_loss, 'lr:', lr,'\n')
            pbar = tqdm(total=len(val_dataloader), desc=f'Epoch {epoch}/{epochs} val')

        #----------------------------------------#
        #   eval
        #----------------------------------------#
        model.eval()
        val_loss_arr = []
        acc_arr = []
        for image, target in val_dataloader:
            if cuda:
                image, target = image.cuda(local_rank), target.cuda(local_rank)

            with torch.inference_mode():
                # 预测
                predict: Tensor = model(image)
                # loss
                loss: Tensor    = loss_fn(predict, target)
            val_loss_arr.append(loss.item())

            # acc,注意取最大值
            pred = predict.argmax(dim=1)
            acc  = pred.eq(target).float().sum().item() / len(image) * 100
            acc_arr.append(acc)     # 这里如果 +=pred.eq(target).float().sum().item() 准确率会降低到50%，不知道怎么计算的

            if local_rank == 0:
                pbar.update(1)
                pbar.set_postfix(**{'loss': loss.item(),
                                    'acc': acc})

        if local_rank == 0:
            pbar.close()
            val_avg_loss = torch.as_tensor(val_loss_arr).mean().item()
            val_avg_acc  = torch.as_tensor(acc_arr).mean().item()
            print(f'Epoch {epoch}/{epochs}', 'val_avg_loss:', val_avg_loss, 'val_avg_acc:', val_avg_acc, '\n')

            #----------------------------------------#
            #   save model
            #----------------------------------------#
            # save_interval 保存一次
            if epoch % save_interval == 0:
                torch.save(model.state_dict(),     f"{save_path}/{str(epoch)}_{str(val_avg_acc)}_model.pth")
                torch.save(optimizer.state_dict(), f"{save_path}/{str(epoch)}_{str(val_avg_acc)}_optim.pth")

            # 保存最好的模型
            if val_avg_acc >= BEST_ACC:
                BEST_ACC = val_avg_acc
                print(f'Epoch {epoch}/{epochs}', 'BEST_ACC:', BEST_ACC, '\n')
                torch.save(model.state_dict(),     f"{save_path}/best_model.pth")
                torch.save(optimizer.state_dict(), f"{save_path}/best_optim.pth")

            end = time.time()

            #----------------------------------------#
            #   记录数据
            #----------------------------------------#
            # tensorboard
            tb_writer.add_scalar('learn_rate',  lr,             epoch)
            tb_writer.add_scalar('train/loss',  train_avg_loss, epoch)
            tb_writer.add_scalar('val/loss',    val_avg_loss,   epoch)
            tb_writer.add_scalar('val/acc',     val_avg_acc,    epoch)
            # csv
            record(epoch, lr, train_avg_loss, val_avg_loss, val_avg_acc, BEST_ACC, int(end-start),
                    time.strftime("%Y%m%d-%H%M%S", time.localtime()), file=f"{save_path}/records.csv")
