import torch
from torch import optim
from torchvision import models
import random
import time
import os


# -------------------------------------#
#   创建需要的文件夹
# -------------------------------------#
def check_dir_exist(dirs=["checkpoint"]):
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
            print(f"dir: '{dir}' created successfully.")


# -------------------------------------#
#   记录数据
# -------------------------------------#
def record(
    epoch,
    lr,
    train_loss,
    val_loss,
    val_acc,
    best_acc,
    times,
    time_now=time.strftime("%Y%m%d-%H%M%S", time.localtime()),
    file="checkpoint/records.csv",
):
    """记录数据

    Args:
        epoch (int):         当前epoch，从1开始
        lr (float):          学习率
        train_loss (float):  训练平均损失
        val_loss (float):    验证平均损失
        val_acc (float):     验证平均准确率
        best_acc (float):    验证最好准确率
        times (int):         训练所需时间
        time_now (time, optional): 当前时间. Defaults to time.strftime("%Y%m%d-%H%M%S", time.localtime()).
        file (str):          保存文件路径. Defaults to checkpoint/records.csv.
    """
    with open(file, mode="a", encoding="utf-8") as f:
        s = (
            str(epoch)
            + ","
            + str(lr)
            + ","
            + str(train_loss)
            + ","
            + str(val_loss)
            + ","
            + str(val_acc)
            + ","
            + str(best_acc)
            + ","
            + str(times)
            + ","
            + time_now
            + "\n"
        )
        if epoch == 1:
            s = "epoch,lr,train_loss,val_loss,val_acc,best_acc,times,time_now\n" + s
        f.write(s)


def test_record():
    for i in range(100):
        time.sleep(0.1)
        print(i)
        record(
            i + 1,
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.random(),
            random.randint(100, 200),
            time.strftime("%Y%m%d-%H%M%S", time.localtime()),
            "../checkpoints/records.csv",
        )


# -------------------------------------#
#   显示config
# -------------------------------------#
def show_config(**kwargs):
    string = "Configurations:\n"
    string += "-" * 180
    string += "\n|%20s | %155s|\n" % ("keys", "values")
    string += "-" * 180
    string += "\n"
    for key, value in kwargs.items():
        string += "|%20s | %155s|\n" % (str(key), str(value))
    string += "-" * 180
    string += "\n"
    # 打印配置
    print(string)
    # 保存配置到文件
    config = os.path.join(kwargs["save_path"], "train.config")
    with open(config, mode="w", encoding="utf-8") as f:
        f.write(string)


def test_show_config():
    import torch
    from torch import nn, optim
    from torchvision import models

    save_path = f"./"
    epochs = 100
    save_interval = 10
    model = models.shufflenet_v2_x1_0()
    optimizer = optim.AdamW(model.parameters())
    lr_sche = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=0.001 * 0.01
    )  # 最小减小100倍
    loss_fn = nn.CrossEntropyLoss()
    device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    half = True
    distributed = False
    sync_bn = False

    # 优化器参数
    optim_param = [
        k + "=" + str(v)
        for k, v in optimizer.state_dict()["param_groups"][0].items()
        if "params" not in k
    ]
    # 学习率降低参数
    if lr_sche:
        lr_param = [
            k + "=" + str(v)
            for k, v in lr_sche.state_dict().items()
            if "params" not in k
        ]
    show_config(
        epochs=epochs,
        save_interval=save_interval,
        device=device,
        model=model.__class__.__name__,
        optimizer=optimizer.__class__.__name__ + ": " + " ".join(optim_param),
        lr_sche=lr_sche.__class__.__name__ + ": " + " ".join(lr_param)
        if lr_sche
        else None,
        loss_fn=loss_fn.__class__.__name__,
        batch_size=batch_size,
        half=half,
        distributed=distributed,
        sync_bn=sync_bn,
        save_path=save_path,
    )
    # Configurations:
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # |                keys |                                                                                                                                                      values|
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # |              epochs |                                                                                                                                                         100|
    # |       save_interval |                                                                                                                                                          10|
    # |              device |                                                                                                                                                      cuda:0|
    # |               model |                                                                                                                                                ShuffleNetV2|
    # |           optimizer |                                                AdamW: lr=0.001 betas=(0.9, 0.999) eps=1e-08 weight_decay=0.01 amsgrad=False maximize=False initial_lr=0.001|
    # |             lr_sche |      CosineAnnealingLR: T_max=100 eta_min=1e-05 base_lrs=[0.001] last_epoch=0 _step_count=1 verbose=False _get_lr_called_within_step=False _last_lr=[0.001]|
    # |             loss_fn |                                                                                                                                            CrossEntropyLoss|
    # |          batch_size |                                                                                                                                                         128|
    # |                half |                                                                                                                                                        True|
    # |         distributed |                                                                                                                                                       False|
    # |             sync_bn |                                                                                                                                                       False|
    # |           save_path |                                                                                                                                                          ./|
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def get_optimizer(name: str, model, lr: float, **kwargs):
    """获取优化器

    Args:
        name (str):          优化器名字
        model (Module):      模型
        lr (float):          学习率

    Returns:
        _type_: _description_
    """
    name = name.lower()
    assert name in [
        "sgd",
        "adam",
        "adamw",
        "adagrad",
        "adadelta",
        "rmsprop",
    ], "optimizer name error"
    print(name)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr, **kwargs)
    elif name == "adam":
        return optim.Adam(model.parameters(), lr, **kwargs)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr, **kwargs)
    elif name == "adagrad":
        return optim.Adagrad(model.parameters(), lr, **kwargs)
    elif name == "adadelta":
        return optim.Adadelta(model.parameters(), lr, **kwargs)
    elif name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr, **kwargs)
    else:
        raise ValueError("optimizer name error")

    # 这样写的话使用**kwargs模型会找到sgd,原因不明
    # return {'sgd':      optim.SGD(model.parameters(), lr, **kwargs),
    #         'adam':     optim.Adam(model.parameters(), lr, **kwargs),
    #         'adamw':    optim.AdamW(model.parameters(), lr, **kwargs),
    #         'adagrad':  optim.Adagrad(model.parameters(), lr, **kwargs),
    #         'adadelta': optim.Adadelta(model.parameters(), lr, **kwargs),
    #         'rmsprop':  optim.RMSprop(model.parameters(), lr, **kwargs)}[name]


def test_get_optimizer():
    model = models.resnet18()
    optimizer = get_optimizer("adam", model, 0.002, eps=1e-5)
    print(optimizer)


def save_file(file_name: str, save_dir: str):
    """保存文件内容

    Args:
        file_name (str): 源文件名称
        save_dir (str): 保存路径
    """
    # 保存路径不存在
    if not os.path.exists(save_dir):
        # 递归创建文件夹
        os.makedirs(save_dir)

    with open(file_name, "r", encoding="utf-8") as f:
        txt = f.read()

    with open(os.path.join(save_dir, file_name), "w", encoding="utf-8") as f:
        f.write(txt)


if __name__ == "__main__":
    # test_record()
    # test_show_config()
    test_get_optimizer()
