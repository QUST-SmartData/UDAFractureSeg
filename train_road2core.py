import argparse
import os
import os.path as osp
from itertools import cycle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.losses import DiceLoss, FocalLoss
from torch.autograd import Variable
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from dataset.core_dataset import CoreDataSet
from dataset.road_dataset import RoadDataSet
from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from model.msaunet import MSAUNet
from model.udaunet import UDAUNet
from utils.loss import CrossEntropy2d

# SOURCE_DATASET = 'Pavements'
SOURCE_DATASET = 'CFD'
# SOURCE_DATASET = 'Cracktree200'
# SOURCE_DATASET = 'DeepCrack'
# SOURCE_DATASET = 'CRACK500'

TARGET_DATASET = 'Carbonate-rich Shale'
# TARGET_DATASET = 'Mancos Shale'

# MODEL = 'DeeplabMulti'
MODEL = 'UDAUNet'
# MODEL = 'MSAUNet'

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = f'/opt/data/private/datasets/paper_data/{SOURCE_DATASET}/'
DATA_LIST_PATH = f'/opt/data/private/datasets/paper_data/{SOURCE_DATASET}/train.txt'
INPUT_SIZE = '512,512'
DATA_DIRECTORY_TARGET = f'/opt/data/private/datasets/paper_data/{TARGET_DATASET}/'
DATA_LIST_PATH_TARGET = f'/opt/data/private/datasets/paper_data/{TARGET_DATASET}/train.txt'
INPUT_SIZE_TARGET = '512,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
IN_CHANNELS = 3
NUM_CLASSES = 2
NUM_STEPS = 400000
NUM_STEPS_STOP = 400000  # early stopping, 10000, 20000, 50000, 300000, 400000
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = ''
# RESTORE_FROM = './DeepLab_resnet_pretrained_init-f81d91e8.pth'
# RESTORE_FROM = f'old_cfd_out/snapshots_{MODEL}_{SOURCE_DATASET}/best_metric_model.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000  # 100, 200, 500, 3000, 4000
SNAPSHOT_DIR = f'snapshots_{MODEL}_{SOURCE_DATASET}_to_{TARGET_DATASET}'
log_dir = f'log_{MODEL}_{SOURCE_DATASET}_to_{TARGET_DATASET}'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
# LAMBDA_SEG = 0
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label1 = label.clone().long().to(gpu)
    label2 = label.clone().unsqueeze(1).to(gpu)
    criterion1 = CrossEntropy2d().to(gpu)
    criterion2 = DiceLoss(to_onehot_y=True, softmax=True)
    criterion3 = FocalLoss(to_onehot_y=True, use_softmax=True)
    loss1 = criterion1(pred, label1)
    loss2 = criterion2(pred, label2)
    loss3 = criterion3(pred, label2)
    loss = loss2
    # loss = loss2 + loss3
    # loss = loss1 + loss2 + loss3

    return loss


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    cudnn.enabled = True
    cudnn.benchmark = True

    os.makedirs(args.snapshot_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    source_dataset = RoadDataSet(args.data_dir, args.data_list, crop_size=input_size)
    trainloader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(cycle(trainloader))

    target_dataset = CoreDataSet(args.data_dir_target, args.data_list_target, crop_size=input_size_target)
    targetloader = data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    targetloader_iter = enumerate(cycle(targetloader))

    # Create network
    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
        optimizer = optim.SGD(model.optim_parameters(args),
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.model == 'UDAUNet':
        model = UDAUNet(IN_CHANNELS, args.num_classes)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    else:
        model = MSAUNet(IN_CHANNELS, args.num_classes)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)

    if args.restore_from:
        model.load_state_dict(torch.load(args.restore_from), strict=False)
        print('loading pretrained model success !')

    model.train()
    model.cuda(args.gpu)

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    model_D1.train()
    model_D1.cuda(args.gpu)

    model_D2.train()
    model_D2.cuda(args.gpu)

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
#         adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
#         adjust_learning_rate_D(optimizer_D1, i_iter)
#         adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # ------- train G -------

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # --- train with source ---
            _, batch = next(trainloader_iter)
            images, labels = batch
            images = images.to(args.gpu)

            pred1, pred2 = model(images)
            pred1 = interp(pred1)
            pred2 = interp(pred2)

            # calc loss
            loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

            # proper normalization loss
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += float(loss_seg1.data.cpu()) / args.iter_size
            loss_seg_value2 += float(loss_seg2.data.cpu()) / args.iter_size

            # --- train with target ---
            _, batch = next(targetloader_iter)
            images, _ = batch
            images = images.to(args.gpu)

            pred_target1, pred_target2 = model(images)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            # calc adv loss
            loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(args.gpu))
            loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(args.gpu))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += float(loss_adv_target1.data.cpu()) / args.iter_size
            loss_adv_target_value2 += float(loss_adv_target2.data.cpu()) / args.iter_size

            # ------- train D -------

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            #  --- train with source ---
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            D_out1 = model_D1(F.softmax(pred1, dim=1))
            D_out2 = model_D2(F.softmax(pred2, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(args.gpu))
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += float(loss_D1.data.cpu())
            loss_D_value2 += float(loss_D2.data.cpu())

            #  --- train with target --- 
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(F.softmax(pred_target1, dim=1))
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(args.gpu))
            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += float(loss_D1.data.cpu())
            loss_D_value2 += float(loss_D2.data.cpu())

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1,
                loss_adv_target_value2, loss_D_value1, loss_D_value2))

        writer.add_scalars("loss",
                           {
                               'seg_1_loss': loss_seg_value1,
                               'seg_2_loss': loss_seg_value2,
                               'adv_1_loss': loss_adv_target_value1,
                               'adv_2_loss': loss_adv_target_value2,
                               'd_1_loss': loss_D_value1,
                               'd_2_loss': loss_D_value2
                           },
                           i_iter)

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, 'model_base_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(),
                       osp.join(args.snapshot_dir, 'model_D1_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D2.state_dict(),
                       osp.join(args.snapshot_dir, 'model_D2_' + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'model_base_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'model_D1_' + str(i_iter) + '.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'model_D2_' + str(i_iter) + '.pth'))

    writer.close()


if __name__ == '__main__':
    main()
