import torch
from torch import nn
from torch import Tensor
from torchvision import models
import time
from tqdm import tqdm


def test_model_speed(
    model: nn.Module, input: Tensor, device="cuda:0", eval=False, repeats: int = 1000
):
    model.to(device)
    input = input.to(device)
    if eval:
        model.eval()
    times = []
    for _ in tqdm(range(repeats)):
        if "cuda" in device:
            # CUDA kernel函数是异步的,要加上线程同步函数,等待kernel中所有线程全部执行完毕再执行CPU端后续指令
            torch.cuda.synchronize()
        start = time.time()
        model(input)
        if "cuda" in device:
            torch.cuda.synchronize()
        times.append(time.time() - start)
    return sum(times) / repeats


if __name__ == "__main__":
    model = models.resnet18()
    x = torch.ones((1, 3, 224, 224))
    avg_time = test_model_speed(model, x, eval=True)
    print(avg_time, "second")
    # 224:  0.003574418306350708 second
    # 512:  0.004980533123016357 second
    # 1024: 0.012861243486404418 second
