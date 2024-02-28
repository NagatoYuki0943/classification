import torch
from torchvision import models
from utils.train_lightning import load_lightning_model


NUM_CLASSES = 6
path = "./checkpoints/scenery/20221110-080100-mobilenet_v3_large-adamw/lightning_logs/version_0/checkpoints/last.ckpt"


if __name__ == "__main__":
    # 模型,因为LightningTrainer需要这个参数初始化，所以必须给
    model = models.mobilenet_v3_large(num_classes=NUM_CLASSES)

    lightning_model = load_lightning_model(model, path)
    model = lightning_model.model
    x = torch.ones(1, 3, 224, 224)

    lightning_model.eval()
    with torch.inference_mode():
        y1 = lightning_model(x)
        y2 = model(x)
    print(y1.size()) # torch.Size([1, 100])
    print(y2.size()) # torch.Size([1, 100])
