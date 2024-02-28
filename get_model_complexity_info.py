from torchvision import models
from mmengine.analysis import get_model_complexity_info


if __name__ == "__main__":
    # model
    model = models.resnet18()
    input_shape = (3, 224, 224)
    analysis_results = get_model_complexity_info(model, input_shape)
    print(analysis_results["out_table"])
    # +------------------------+----------------------+------------+--------------+
    # | module                 | #parameters or shape | #flops     | #activations |
    # +------------------------+----------------------+------------+--------------+
    # | model                  | 11.69M               | 1.827G     | 2.485M       |
    # |  conv1                 |  9.408K              |  0.118G    |  0.803M      |
    # |   conv1.weight         |   (64, 3, 7, 7)      |            |              |
    # |  bn1                   |  0.128K              |  4.014M    |  0           |
    # |   bn1.weight           |   (64,)              |            |              |
    # |   bn1.bias             |   (64,)              |            |              |
    # |  layer1                |  0.148M              |  0.466G    |  0.803M      |
    # |   layer1.0             |   73.984K            |   0.233G   |   0.401M     |
    # |    layer1.0.conv1      |    36.864K           |    0.116G  |    0.201M    |
    # |    layer1.0.bn1        |    0.128K            |    1.004M  |    0         |
    # |    layer1.0.conv2      |    36.864K           |    0.116G  |    0.201M    |
    # |    layer1.0.bn2        |    0.128K            |    1.004M  |    0         |
    # |   layer1.1             |   73.984K            |   0.233G   |   0.401M     |
    # |    layer1.1.conv1      |    36.864K           |    0.116G  |    0.201M    |
    # |    layer1.1.bn1        |    0.128K            |    1.004M  |    0         |
    # |    layer1.1.conv2      |    36.864K           |    0.116G  |    0.201M    |
    # |    layer1.1.bn2        |    0.128K            |    1.004M  |    0         |
    # |  layer2                |  0.526M              |  0.414G    |  0.502M      |
    # |   layer2.0             |   0.23M              |   0.181G   |   0.301M     |
    # |    layer2.0.conv1      |    73.728K           |    57.803M |    0.1M      |
    # |    layer2.0.bn1        |    0.256K            |    0.502M  |    0         |
    # |    layer2.0.conv2      |    0.147M            |    0.116G  |    0.1M      |
    # |    layer2.0.bn2        |    0.256K            |    0.502M  |    0         |
    # |    layer2.0.downsample |    8.448K            |    6.924M  |    0.1M      |
    # |   layer2.1             |   0.295M             |   0.232G   |   0.201M     |
    # |    layer2.1.conv1      |    0.147M            |    0.116G  |    0.1M      |
    # |    layer2.1.bn1        |    0.256K            |    0.502M  |    0         |
    # |    layer2.1.conv2      |    0.147M            |    0.116G  |    0.1M      |
    # |    layer2.1.bn2        |    0.256K            |    0.502M  |    0         |
    # |  layer3                |  2.1M                |  0.412G    |  0.251M      |
    # |   layer3.0             |   0.919M             |   0.181G   |   0.151M     |
    # |    layer3.0.conv1      |    0.295M            |    57.803M |    50.176K   |
    # |    layer3.0.bn1        |    0.512K            |    0.251M  |    0         |
    # |    layer3.0.conv2      |    0.59M             |    0.116G  |    50.176K   |
    # |    layer3.0.bn2        |    0.512K            |    0.251M  |    0         |
    # |    layer3.0.downsample |    33.28K            |    6.673M  |    50.176K   |
    # |   layer3.1             |   1.181M             |   0.232G   |   0.1M       |
    # |    layer3.1.conv1      |    0.59M             |    0.116G  |    50.176K   |
    # |    layer3.1.bn1        |    0.512K            |    0.251M  |    0         |
    # |    layer3.1.conv2      |    0.59M             |    0.116G  |    50.176K   |
    # |    layer3.1.bn2        |    0.512K            |    0.251M  |    0         |
    # |  layer4                |  8.394M              |  0.412G    |  0.125M      |
    # |   layer4.0             |   3.673M             |   0.18G    |   75.264K    |
    # |    layer4.0.conv1      |    1.18M             |    57.803M |    25.088K   |
    # |    layer4.0.bn1        |    1.024K            |    0.125M  |    0         |
    # |    layer4.0.conv2      |    2.359M            |    0.116G  |    25.088K   |
    # |    layer4.0.bn2        |    1.024K            |    0.125M  |    0         |
    # |    layer4.0.downsample |    0.132M            |    6.548M  |    25.088K   |
    # |   layer4.1             |   4.721M             |   0.231G   |   50.176K    |
    # |    layer4.1.conv1      |    2.359M            |    0.116G  |    25.088K   |
    # |    layer4.1.bn1        |    1.024K            |    0.125M  |    0         |
    # |    layer4.1.conv2      |    2.359M            |    0.116G  |    25.088K   |
    # |    layer4.1.bn2        |    1.024K            |    0.125M  |    0         |
    # |  fc                    |  0.513M              |  0.512M    |  1K          |
    # |   fc.weight            |   (1000, 512)        |            |              |
    # |   fc.bias              |   (1000,)            |            |              |
    # |  avgpool               |                      |  25.088K   |  0           |
    # +------------------------+----------------------+------------+--------------+

    print(analysis_results["out_arch"])
    # ResNet(
    #   #params: 11.69M, #flops: 1.83G, #acts: 2.48M
    #   (conv1): Conv2d(
    #     3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    #     #params: 9.41K, #flops: 0.12G, #acts: 0.8M
    #   )
    #   (bn1): BatchNorm2d(
    #     64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #     #params: 0.13K, #flops: 4.01M, #acts: 0
    #   )
    #   (relu): ReLU(inplace=True)
    #   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    #   (layer1): Sequential(
    #     #params: 0.15M, #flops: 0.47G, #acts: 0.8M
    #     (0): BasicBlock(
    #       #params: 73.98K, #flops: 0.23G, #acts: 0.4M
    #       (conv1): Conv2d(
    #         64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 36.86K, #flops: 0.12G, #acts: 0.2M
    #       )
    #       (bn1): BatchNorm2d(
    #         64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.13K, #flops: 1M, #acts: 0
    #       )
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(
    #         64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 36.86K, #flops: 0.12G, #acts: 0.2M
    #       )
    #       (bn2): BatchNorm2d(
    #         64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.13K, #flops: 1M, #acts: 0
    #       )
    #     )
    #     (1): BasicBlock(
    #       #params: 73.98K, #flops: 0.23G, #acts: 0.4M
    #       (conv1): Conv2d(
    #         64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 36.86K, #flops: 0.12G, #acts: 0.2M
    #       )
    #       (bn1): BatchNorm2d(
    #         64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.13K, #flops: 1M, #acts: 0
    #       )
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(
    #         64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 36.86K, #flops: 0.12G, #acts: 0.2M
    #       )
    #       (bn2): BatchNorm2d(
    #         64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.13K, #flops: 1M, #acts: 0
    #       )
    #     )
    #   )
    #   (layer2): Sequential(
    #     #params: 0.53M, #flops: 0.41G, #acts: 0.5M
    #     (0): BasicBlock(
    #       #params: 0.23M, #flops: 0.18G, #acts: 0.3M
    #       (conv1): Conv2d(
    #         64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    #         #params: 73.73K, #flops: 57.8M, #acts: 0.1M
    #       )
    #       (bn1): BatchNorm2d(
    #         128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.26K, #flops: 0.5M, #acts: 0
    #       )
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(
    #         128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 0.15M, #flops: 0.12G, #acts: 0.1M
    #       )
    #       (bn2): BatchNorm2d(
    #         128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.26K, #flops: 0.5M, #acts: 0
    #       )
    #       (downsample): Sequential(
    #         #params: 8.45K, #flops: 6.92M, #acts: 0.1M
    #         (0): Conv2d(
    #           64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False
    #           #params: 8.19K, #flops: 6.42M, #acts: 0.1M
    #         )
    #         (1): BatchNorm2d(
    #           128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #           #params: 0.26K, #flops: 0.5M, #acts: 0
    #         )
    #       )
    #     )
    #     (1): BasicBlock(
    #       #params: 0.3M, #flops: 0.23G, #acts: 0.2M
    #       (conv1): Conv2d(
    #         128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 0.15M, #flops: 0.12G, #acts: 0.1M
    #       )
    #       (bn1): BatchNorm2d(
    #         128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.26K, #flops: 0.5M, #acts: 0
    #       )
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(
    #         128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 0.15M, #flops: 0.12G, #acts: 0.1M
    #       )
    #       (bn2): BatchNorm2d(
    #         128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.26K, #flops: 0.5M, #acts: 0
    #       )
    #     )
    #   )
    #   (layer3): Sequential(
    #     #params: 2.1M, #flops: 0.41G, #acts: 0.25M
    #     (0): BasicBlock(
    #       #params: 0.92M, #flops: 0.18G, #acts: 0.15M
    #       (conv1): Conv2d(
    #         128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    #         #params: 0.29M, #flops: 57.8M, #acts: 50.18K
    #       )
    #       (bn1): BatchNorm2d(
    #         256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.51K, #flops: 0.25M, #acts: 0
    #       )
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(
    #         256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 0.59M, #flops: 0.12G, #acts: 50.18K
    #       )
    #       (bn2): BatchNorm2d(
    #         256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.51K, #flops: 0.25M, #acts: 0
    #       )
    #       (downsample): Sequential(
    #         #params: 33.28K, #flops: 6.67M, #acts: 50.18K
    #         (0): Conv2d(
    #           128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False
    #           #params: 32.77K, #flops: 6.42M, #acts: 50.18K
    #         )
    #         (1): BatchNorm2d(
    #           256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #           #params: 0.51K, #flops: 0.25M, #acts: 0
    #         )
    #       )
    #     )
    #     (1): BasicBlock(
    #       #params: 1.18M, #flops: 0.23G, #acts: 0.1M
    #       (conv1): Conv2d(
    #         256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 0.59M, #flops: 0.12G, #acts: 50.18K
    #       )
    #       (bn1): BatchNorm2d(
    #         256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.51K, #flops: 0.25M, #acts: 0
    #       )
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(
    #         256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 0.59M, #flops: 0.12G, #acts: 50.18K
    #       )
    #       (bn2): BatchNorm2d(
    #         256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 0.51K, #flops: 0.25M, #acts: 0
    #       )
    #     )
    #   )
    #   (layer4): Sequential(
    #     #params: 8.39M, #flops: 0.41G, #acts: 0.13M
    #     (0): BasicBlock(
    #       #params: 3.67M, #flops: 0.18G, #acts: 75.26K
    #       (conv1): Conv2d(
    #         256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    #         #params: 1.18M, #flops: 57.8M, #acts: 25.09K
    #       )
    #       (bn1): BatchNorm2d(
    #         512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 1.02K, #flops: 0.13M, #acts: 0
    #       )
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(
    #         512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 2.36M, #flops: 0.12G, #acts: 25.09K
    #       )
    #       (bn2): BatchNorm2d(
    #         512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 1.02K, #flops: 0.13M, #acts: 0
    #       )
    #       (downsample): Sequential(
    #         #params: 0.13M, #flops: 6.55M, #acts: 25.09K
    #         (0): Conv2d(
    #           256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False
    #           #params: 0.13M, #flops: 6.42M, #acts: 25.09K
    #         )
    #         (1): BatchNorm2d(
    #           512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #           #params: 1.02K, #flops: 0.13M, #acts: 0
    #         )
    #       )
    #     )
    #     (1): BasicBlock(
    #       #params: 4.72M, #flops: 0.23G, #acts: 50.18K
    #       (conv1): Conv2d(
    #         512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 2.36M, #flops: 0.12G, #acts: 25.09K
    #       )
    #       (bn1): BatchNorm2d(
    #         512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 1.02K, #flops: 0.13M, #acts: 0
    #       )
    #       (relu): ReLU(inplace=True)
    #       (conv2): Conv2d(
    #         512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    #         #params: 2.36M, #flops: 0.12G, #acts: 25.09K
    #       )
    #       (bn2): BatchNorm2d(
    #         512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
    #         #params: 1.02K, #flops: 0.13M, #acts: 0
    #       )
    #     )
    #   )
    #   (avgpool): AdaptiveAvgPool2d(
    #     output_size=(1, 1)
    #     #params: 0, #flops: 25.09K, #acts: 0
    #   )
    #   (fc): Linear(
    #     in_features=512, out_features=1000, bias=True
    #     #params: 0.51M, #flops: 0.51M, #acts: 1K
    #   )
    # )

    print("Model Flops:{}".format(analysis_results["flops_str"]))
    # Model Flops:1.827G

    print("Model Parameters:{}".format(analysis_results["params_str"]))
    # Model Parameters:11.69M
