import numpy as np
from PIL import Image
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


SOURCE_DATASET = 'road'
# SOURCE_DATASET = 'cfd'
TARGET_DATASET = 'core'

# model_name ='DeeplabMulti'
model_name = 'MSAUNet'
# model_name = 'UNet'

# uda
# pred_path = f"pred_{model_name}_{SOURCE_DATASET}_to_{TARGET_DATASET}"

# s-o
# pred_path = f"pred_{model_name}_{SOURCE_DATASET}"

label_path = f'/opt/data/private/datasets/fracture/{TARGET_DATASET}/test/labels'


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


if __name__ == '__main__':
    pred_files = os.listdir(pred_path)
    label_files = os.listdir(label_path)

    pred_files = [file for file in pred_files if os.path.isfile(os.path.join(pred_path, file))]
    label_files = [file for file in label_files if os.path.isfile(os.path.join(label_path, file))]

    y_true = []
    y_pred = []
    miou = 0
    tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0

    for pred_file, label_file in zip(pred_files, label_files):
        label_file_path = os.path.join(label_path, label_file)
        pred_file_path = os.path.join(pred_path, pred_file)

        label_image = Image.open(label_file_path).convert('L')
        pred_image = Image.open(pred_file_path).convert('L')

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
