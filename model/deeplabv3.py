from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50


class Deeplab_v3(nn.Module):
    def __init__(self, weights=False, num_classes=2, weights_backbone=False):
        super(Deeplab_v3, self).__init__()
        self.model1 = deeplabv3_resnet50(
            weights=weights,
            num_classes=num_classes,
            weights_backbone=weights_backbone
        )

    def forward(self, input):
        x = self.model1(input)['out']
        return x
