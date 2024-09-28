import os.path
from glob import glob

import numpy as np
import torch

from PIL import Image
import os

from PIL.Image import Resampling
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from monai.networks.nets import UNet

from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Subset

from torch import nn
from model.deeplab_multi import DeeplabMulti
from dataset.core_dataset import CoreDataSet
from model.deeplabv3 import Deeplab_v3
from model.msaunet import MSAUNet
from model.udaunet import UDAUNet
from model.pspnet import PSPNet
from model.segnet import SegNet
from model.deeplab_multi import DeeplabMulti

# SOURCE_DATASET = 'Pavements'
SOURCE_DATASET = 'CFD'
# SOURCE_DATASET = 'Cracktree200'
# SOURCE_DATASET = 'DeepCrack'
# SOURCE_DATASET = 'CRACK500'

# TARGET_DATASET = 'Carbonate-rich Shale'
TARGET_DATASET = 'Mancos Shale'

MODEL = 'DeeplabMulti'
# MODEL = 'UDAUNet'
# MODEL = 'MSAUNet'

# MODEL = 'UNet'
# MODEL = 'SegNet'
# MODEL = 'PSPNet'
# MODEL = 'Deeplabv3'

in_channels = 3
num_classes = 2
crop_size = (512, 512)
out_size = (512, 512)
test_dataset_dir = f'/opt/data/private/datasets/fracture/{TARGET_DATASET}/test'
test_dataset_list = f'/opt/data/private/datasets/fracture/{TARGET_DATASET}_test_list.txt'

# uda
# output_dir = f"pred_{MODEL}_{SOURCE_DATASET}_to_{TARGET_DATASET}"
# model_saved_path = f'snapshots_{MODEL}_{SOURCE_DATASET}_to_{TARGET_DATASET}/model_base_85000.pth'

# s-o
# output_dir = f"pred_{MODEL}_{SOURCE_DATASET}"
# model_saved_path = f'snapshots_{MODEL}_{SOURCE_DATASET}/best_metric_model.pth'

output_dir = f"batch_pred_{MODEL}_{SOURCE_DATASET}"

model_saved_path = sorted(glob(f'old_cfd_pt_out/snapshots_{MODEL}_{SOURCE_DATASET}/*.pth'))


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

label_path = f'/opt/data/private/datasets/fracture/{TARGET_DATASET}/test/labels'

ml1,ml2 = [], []

def mean_iou(target, pred, classes=2):
    """  compute the value of mean iou
    :param pred:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    for i in range(classes):
        intersection = np.logical_and(target == i, pred == i)
        # print(intersection.any())
        union = np.logical_or(target == i, pred == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return miou / classes


def calc_m(out_dir):
    pred_files = sorted(glob(f'{out_dir}/*.png'))
    label_files = sorted(glob(f'{label_path}/*.png'))
    
    print('-'*10)
    ml1.append(calc(pred_files[:len(pred_files)//2], label_files[:len(pred_files)//2]))
    print('-'*10)
    ml2.append(calc(pred_files[len(pred_files)//2:], label_files[len(pred_files)//2:]))


def calc(pred_files, label_files):
    y_true = []
    y_pred = []
    miou = 0
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0

    for pred_file, label_file in zip(pred_files, label_files):

        label_image = Image.open(label_file).convert('L')
        pred_image = Image.open(pred_file).convert('L')

        label_array = np.array(label_image) / 255
        pred_array = np.array(pred_image) / 255

        label_array = label_array.astype(np.uint8)
        pred_array = pred_array.astype(np.uint8)

        miou += mean_iou(label_array, pred_array)

        y_true.extend(label_array.flatten())
        y_pred.extend(pred_array.flatten())
        
        # Calculate TP, TN, FP, FN
        tp = np.sum((label_array == 1) & (pred_array == 1))
        tn = np.sum((label_array == 0) & (pred_array == 0))
        fp = np.sum((label_array == 0) & (pred_array == 1))
        fn = np.sum((label_array == 1) & (pred_array == 0))

        tp_total += tp
        tn_total += tn
        fp_total += fp
        fn_total += fn

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    miou /= len(pred_files)
    
    
    print(f'TP: {tp_total}')
    print(f'TN: {tn_total}')
    print(f'FP: {fp_total}')
    print(f'FN: {fn_total}')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'MIoU: {miou:.4f}')

    return accuracy,precision,recall,f1,miou



def main():
    

    test_dataset = CoreDataSet(test_dataset_dir, test_dataset_list, crop_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    for mp in model_saved_path:
        print('-'*20)
        print(mp)
        
        pred_img_out_dir = f"{output_dir}/{os.path.basename(mp)}"
        os.makedirs(pred_img_out_dir, exist_ok=True)

        model.load_state_dict(torch.load(mp))
        
        interp = nn.Upsample(size=crop_size, mode='bilinear')

        model.eval()
        with torch.no_grad():
            for data in test_loader:
                images = data[0].to(device)
                outputs = model(images)
                outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
                outputs = interp(outputs)
                outputs = softmax(outputs, dim=1)
                for output in outputs:
                    out_image = torch.argmax(output, dim=0).to(torch.uint8) * 255
                    image = Image.fromarray(np.asarray(out_image.cpu()))
                    image = image.resize(out_size, Resampling.NEAREST)
                    image.save(os.path.join(pred_img_out_dir, str(data[-1][0])))

        calc_m(pred_img_out_dir)



if __name__ == "__main__":
    main()

    data1 = np.asarray(ml1)*100
    np.savetxt(f'{output_dir}/metrics-1.csv', data1, delimiter=',', fmt='%.2f')

    data2 = np.asarray(ml2)*100
    np.savetxt(f'{output_dir}/metrics-2.csv', data2, delimiter=',', fmt='%.2f')