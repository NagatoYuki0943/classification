def _make_divisible(v: float, divisor: int=8, min_value: int | None = None) -> int:
    """
    将卷积核个数(输出通道个数)调整为最接近round_nearest的整数倍,就是8的整数倍,对硬件更加友好
    v:          输出通道个数
    divisor:    奇数,必须将ch调整为它的整数倍
    min_value:  最小通道数

    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


if __name__ == "__main__":
    print(_make_divisible(6, 8))    # 8
    print(_make_divisible(7, 8))    # 8
    print(_make_divisible(8, 8))    # 8
    print(_make_divisible(10, 8))   # 16
    print(_make_divisible(12, 8))   # 16
    print(_make_divisible(14, 8))   # 16