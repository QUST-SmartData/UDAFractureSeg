import os.path

import numpy as np
import torch

from PIL import Image
from PIL.Image import Resampling

from monai.networks.nets import UNet

from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from torch import nn
from model.deeplab_multi import DeeplabMulti
from dataset.core_dataset import CoreDataSet
from model.deeplabv3 import Deeplab_v3
from model.msaunet import MSAUNet
from model.udaunet import UDAUNet
from model.pspnet import PSPNet
from model.segnet import SegNet
from model.deeplab_multi import DeeplabMulti

SOURCE_DATASET = 'Pavements'
# SOURCE_DATASET = 'CFD'
# SOURCE_DATASET = 'Cracktree200'
# SOURCE_DATASET = 'DeepCrack'
# SOURCE_DATASET = 'CRACK500'

# TARGET_DATASET = 'Carbonate-rich Shale'
TARGET_DATASET = 'Mancos Shale'

# MODEL = 'DeeplabMulti'
# MODEL = 'UDAUNet'
# MODEL = 'MSAUNet'

# MODEL = 'UNet'
# MODEL = 'SegNet'
# MODEL = 'PSPNet'
MODEL = 'Deeplabv3'

in_channels = 3
num_classes = 2
crop_size = (512, 512)
out_size = (512, 512)
test_dataset_dir = f'/opt/data/private/datasets/paper_data/{TARGET_DATASET}/'
test_dataset_list = f'/opt/data/private/datasets/paper_data/{TARGET_DATASET}/test.txt'

# uda
# output_dir = f"pred_{MODEL}_{SOURCE_DATASET}_to_{TARGET_DATASET}"
# model_saved_path = f'snapshots_{MODEL}_{SOURCE_DATASET}_to_{TARGET_DATASET}/model_base_800.pth'

# s-o
output_dir = f"pred_{MODEL}_{SOURCE_DATASET}_{TARGET_DATASET}"
model_saved_path = f'snapshots_{MODEL}_{SOURCE_DATASET}/best_metric_model.pth'

# output_dir = f"pred_{MODEL}_{SOURCE_DATASET}"
# model_saved_path = f'old_cfd_out/snapshots_{MODEL}_{SOURCE_DATASET}/best_metric_model.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL == "UNet":
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
elif MODEL == "SegNet":
    model = SegNet(
        in_chn=in_channels,
        out_chn=num_classes
    )
elif MODEL == "Deeplabv3":
    model = Deeplab_v3(num_classes=num_classes)
elif MODEL == "PSPNet":
    model = PSPNet(classes=num_classes)
elif MODEL == 'DeeplabMulti':
    model = DeeplabMulti(num_classes=num_classes)
elif MODEL == 'UDAUNet':
    model = UDAUNet(in_channels, num_classes)
else:
    model = MSAUNet(
        in_channels=in_channels,
        out_channels=num_classes,
    )
model.to(device)


def main():
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = CoreDataSet(test_dataset_dir, test_dataset_list, crop_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model.load_state_dict(torch.load(model_saved_path))
    
    interp = nn.Upsample(size=crop_size, mode='bilinear')

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images = data[0].to(device)
            outputs = model(images)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = interp(outputs)
            for output in outputs:
                out_image = torch.argmax(output, dim=0).to(torch.uint8) * 255
                image = Image.fromarray(np.asarray(out_image.cpu()))
                image = image.resize(out_size, Resampling.NEAREST)
                image.save(os.path.join(output_dir, str(data[-1][0])))


if __name__ == "__main__":
    main()
