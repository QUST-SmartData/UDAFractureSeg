import os
import random
import shutil

import torch
from monai.losses import DiceLoss, FocalLoss
from monai.networks import one_hot
from monai.networks.nets import UNet
from torch.nn.functional import softmax

from torch import nn
# from torch.nn import CrossEntropyLoss
from utils.loss import CrossEntropy2d
from torch.utils import data
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from monai.metrics import DiceMetric

from dataset.road_dataset import RoadDataSet
from model.deeplabv3 import Deeplab_v3
from model.deeplab_multi import DeeplabMulti
from model.msaunet import MSAUNet
from model.udaunet import UDAUNet
from model.pspnet import PSPNet
from model.segnet import SegNet


def main():
    writer = SummaryWriter(log_dir)

    train_dataset = RoadDataSet(data_dir, train_data_list, crop_size)
    valid_dataset = RoadDataSet(data_dir, valid_data_list, crop_size)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=2, pin_memory=True, drop_last=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=2, pin_memory=True, drop_last=True)

    dice_metric = DiceMetric(reduction="mean", get_not_nans=False)

    loss_function1 = DiceLoss(to_onehot_y=True, softmax=True)
    loss_function2 = FocalLoss(to_onehot_y=True, use_softmax=True)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    best_metric = -1
    best_metric_epoch = 0
    epoch_loss_values = list()
    metric_values = list()
    interp = nn.Upsample(size=crop_size, mode='bilinear')
    model_save_path = os.path.join(snapshot_dir, f"{best_metric_epoch}-best_metric_model.pth")

    for epoch in range(epoches):
        print("-" * 20)
        print(f"{model_name} - epoch {epoch + 1}/{epoches}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = interp(outputs)
            loss = loss_function1(outputs, labels)
#             loss = loss_function2(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_dataset) // train_loader.batch_size + len(train_dataset) % train_loader.batch_size
            print(f"{model_name} - {step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"{model_name} - epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in valid_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_labels = val_labels.unsqueeze(1)
                    val_labels = one_hot(val_labels, num_classes)
                    val_outputs = model(val_images)
                    val_outputs = val_outputs[-1] if isinstance(val_outputs, tuple) else val_outputs
                    val_outputs = interp(val_outputs)
                    val_outputs = softmax(val_outputs, dim=1)
                    val_outputs = torch.where(val_outputs >= 0.5, 1, 0)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                
#                 # save every epoch model
#                 model_save_path = os.path.join(snapshot_dir, f"{epoch}_model.pth")
#                 torch.save(model.state_dict(), model_save_path)
                    
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    model_save_path = os.path.join(snapshot_dir, f"{best_metric_epoch}-best_metric_model.pth")
                    torch.save(model.state_dict(), model_save_path)
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)

    shutil.copyfile(model_save_path, best_model_save_path)
    print(f"{model_name} - train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    random.seed(1234)

    # models = ['UNet', 'SegNet', 'PSPNet', 'Deeplabv3', 'UDAUNet', 'MSAUNet', 'DeeplabMulti']
    models = ['UDAUNet', 'MSAUNet', 'DeeplabMulti']

    for m in models:
        model_name = m

        epoches = 100
        lr = 1e-3
        # batch_size>1
        batch_size = 8
        in_channels = 3
        num_classes = 2
        crop_size = (512, 512)
        val_interval = 1

        SOURCE_DATASET = 'Pavements'
        # SOURCE_DATASET = 'CFD'
        # SOURCE_DATASET = 'Cracktree200'
        # SOURCE_DATASET = 'DeepCrack'
        # SOURCE_DATASET = 'CRACK500'
        data_dir = f'/opt/data/private/datasets/paper_data/{SOURCE_DATASET}/'
        train_data_list = f'/opt/data/private/datasets/paper_data/{SOURCE_DATASET}/train.txt'
        valid_data_list = f'/opt/data/private/datasets/paper_data/{SOURCE_DATASET}/valid.txt'

        log_dir = f"log_{model_name}_{SOURCE_DATASET}"
        snapshot_dir = f'snapshots_{model_name}_{SOURCE_DATASET}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)

        best_model_save_path = os.path.join(snapshot_dir, "best_metric_model.pth")

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
        elif model_name == 'UDAUNet':
            model = UDAUNet(in_channels, num_classes)
        elif model_name == 'DeeplabMulti':
            model = DeeplabMulti(num_classes=num_classes)
        else:
            model = MSAUNet(
                in_channels=in_channels,
                out_channels=num_classes,
            )
        model.to(device)

#         model_saved_path = f'old_cfd_out/snapshots_{model_name}_{SOURCE_DATASET}/best_metric_model.pth'
#         if model_saved_path:
#             data_dir = "/opt/data/private/datasets/fracture/core/test"
#             data_list = '/opt/data/private/datasets/fracture/core_test_list.txt'
#             epoches = 10
#             model.load_state_dict(torch.load(model_saved_path))
#             print('loaded pretrained model')

        main()
