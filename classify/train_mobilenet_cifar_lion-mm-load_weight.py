import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class MobileNet(nn.Module):
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


save_path = "checkpoints/mobilenet_v3_large-cifar-lion-mm/best_accuracy_epoch_94.pth"
state_dict: dict = torch.load(save_path)
print(state_dict.keys())    # ['meta', 'state_dict', 'message_hub', 'optimizer', 'param_schedulers', 'ema_state_dict']

model = MobileNet(variant='mobilenet_v3_large', num_classes=10)
model.load_state_dict(state_dict['state_dict'])
