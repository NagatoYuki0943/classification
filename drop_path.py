import torch
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """x为输入的张量，其通道为[B,C,H,W]，那么drop_path的含义为在一个Batch_size中，随机有drop_prob的样本，不经过主干，而直接由分支进行恒等映射
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # [B, 1, 1, 1] 随机将1条数据删除
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor # 除以keep_prob用来保持均值不变
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def test_drop_path():
    x = torch.ones(4, 1, 2, 2)
    print(drop_path(x, 0.5, True))
    # 将某一条数据直接删除
    # tensor([[[[2., 2.],
    #           [2., 2.]]],
    #         [[[0., 0.],
    #           [0., 0.]]],
    #         [[[2., 2.],
    #           [2., 2.]]],
    #         [[[0., 0.],
    #           [0., 0.]]]])


def test_drop_out():
    x = torch.ones(4, 1, 2, 2)
    drop = nn.Dropout(0.5)
    print(drop(x))
    # 随机删除数据,更容易理解
    # tensor([[[[0., 0.],
    #           [2., 0.]]],
    #         [[[0., 0.],
    #           [2., 2.]]],
    #         [[[2., 2.],
    #           [2., 2.]]],
    #         [[[0., 2.],
    #           [2., 0.]]]])


if __name__ == "__main__":
    test_drop_path()
    test_drop_out()

