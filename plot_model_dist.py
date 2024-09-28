import os.path

import numpy as np
import torch

from PIL import Image

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

# SOURCE_DATASET = 'road_t1'
# SOURCE_DATASET = 'road_t2'
# SOURCE_DATASET = 'road'
SOURCE_DATASET = 'cfd'
TARGET_DATASET = 'core'
# TARGET_DATASET = 'voorn'

# model_name ='DeeplabMulti'
# model_name = 'MSAUNet'
model_name = 'UDAUNet'

in_channels = 3
num_classes = 2
crop_size = (512, 512)
out_size = (320, 320)
test_dataset_dir = f'/opt/data/private/datasets/fracture/{TARGET_DATASET}/test'
test_dataset_list = f'/opt/data/private/datasets/fracture/{TARGET_DATASET}_test_list.txt'

# uda
# output_dir = f"pred_{model_name}_{SOURCE_DATASET}_to_{TARGET_DATASET}"
# model_saved_path = f'snapshots_{model_name}_{SOURCE_DATASET}_to_{TARGET_DATASET}/model_base_15000.pth'

# s-o
output_dir = f"pred_{model_name}_{SOURCE_DATASET}"
model_saved_path = f'snapshots_{model_name}_{SOURCE_DATASET}/best_metric_model.pth'

# output_dir = f"pred_{model_name}_{SOURCE_DATASET}"
# model_saved_path = f'old_cfd_out/snapshots_{model_name}_{SOURCE_DATASET}/best_metric_model.pth'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_name == "UNet":
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
elif model_name == "SegNet":
    model = SegNet(
        in_chn=in_channels,
        out_chn=num_classes
    )
elif model_name == "Deeplabv3":
    model = Deeplab_v3(num_classes=num_classes)
elif model_name == "PSPNet":
    model = PSPNet(classes=num_classes)
elif model_name == 'DeeplabMulti':
    model = DeeplabMulti(num_classes=num_classes)
elif model_name == 'UDAUNet':
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
            outputs = softmax(outputs, dim=1)
            for output in outputs:
                out_image = torch.argmax(output, dim=0).to(torch.uint8) * 255
                image = Image.fromarray(np.asarray(out_image.cpu()))
                image = image.resize(out_size, Image.NEAREST)
                image.save(os.path.join(output_dir, str(data[-1][0])))


if __name__ == "__main__":
    main()
