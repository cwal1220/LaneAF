import os
import json
from datetime import datetime
from statistics import mean
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.tusimple import TuSimple
from models.loss import FocalLoss, IoULoss, RegL1Loss
from utils.runtime import build_model, load_snapshot, move_sample_to_device, resolve_device


parser = argparse.ArgumentParser('Options for training LaneAF models in PyTorch...')
parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--backbone', type=str, default='erfnet', help='type of model backbone (dla34/erfnet/enet)')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=40, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='WD', help='weight decay')
parser.add_argument('--loss-type', type=str, default='wbce', help='type of classification loss to use (focal/bce/wbce)')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--device', type=str, default='auto', help='device to use (auto/cpu/cuda/mps)')
parser.add_argument('--random-transforms', action='store_true', default=False, help='apply random transforms to input during training')

args = None
device = None
train_loader = None
val_loader = None
best_f1 = 0.0
f_log = None
tb_writer = None
optimizer = None
scheduler = None
criterion_1 = None
criterion_2 = None
criterion_reg = None
model = None
hm_channel_names = ('ego_left', 'ego', 'ego_right')


def compute_bce_or_focal_loss(output_ch, target_ch, valid_mask):
    if args.loss_type == 'focal':
        return criterion_1(output_ch * valid_mask, target_ch * valid_mask)
    if args.loss_type == 'bce':
        return criterion_1(output_ch * valid_mask, target_ch * valid_mask)
    if args.loss_type == 'wbce':
        pos_weight = torch.tensor([9.6], device=output_ch.device)
        return F.binary_cross_entropy_with_logits(output_ch * valid_mask, target_ch * valid_mask, pos_weight=pos_weight)
    raise AssertionError('Unsupported loss type')


def compute_segmentation_losses(outputs_hm, input_mask, ignore_label):
    channel_losses = {}
    total = None
    for idx, name in enumerate(hm_channel_names):
        output_ch = outputs_hm[:, idx:idx+1, :, :]
        target_ch = input_mask[:, idx:idx+1, :, :]
        valid_mask = (target_ch != ignore_label).float()
        channel_loss = compute_bce_or_focal_loss(output_ch, target_ch, valid_mask) + criterion_2(torch.sigmoid(output_ch), target_ch)
        channel_losses[name] = channel_loss
        total = channel_loss if total is None else total + channel_loss
    return total, channel_losses


def setup():
    global args, device, train_loader, val_loader, best_f1, f_log, tb_writer

    args = parser.parse_args()
    if args.dataset_dir is None:
        raise AssertionError('Path to dataset not provided!')

    args.device = resolve_device(args.device, args.no_cuda)
    device = torch.device(args.device)
    args.cuda = device.type == 'cuda'
    if args.output_dir is None:
        args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M")
        args.output_dir = os.path.join('.', 'experiments', 'tusimple', args.output_dir)

    args.backbone = args.backbone.lower()
    if args.backbone not in ['dla34', 'erfnet', 'enet']:
        raise AssertionError('Incorrect model backbone provided!')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        raise AssertionError('Output directory already exists!')

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 8, 'pin_memory': args.cuda}
    train_loader = DataLoader(TuSimple(args.dataset_dir, 'train', args.random_transforms), **kwargs)
    kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 8, 'pin_memory': args.cuda}
    val_loader = DataLoader(TuSimple(args.dataset_dir, 'val', False), **kwargs)

    best_f1 = 0.0
    f_log = open(os.path.join(args.output_dir, "logs.txt"), "w")
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))


# training function
def train(net, epoch):
    epoch_loss_seg, epoch_loss_vaf, epoch_loss_haf, epoch_loss, epoch_acc, epoch_f1 = list(), list(), list(), list(), list(), list()
    epoch_loss_seg_channels = {name: [] for name in hm_channel_names}
    net.train()
    for b_idx, sample in enumerate(train_loader):
        input_img, input_seg, input_mask, input_af = move_sample_to_device(sample, device)

        # zero gradients before forward pass
        optimizer.zero_grad()

        # do the forward pass
        outputs = net(input_img)[-1]

        # calculate losses and metrics
        af_valid_mask = (input_seg != train_loader.dataset.ignore_label).float()
        loss_seg, loss_seg_channels = compute_segmentation_losses(outputs['hm'], input_mask, train_loader.dataset.ignore_label)
        loss_vaf = 0.5*criterion_reg(outputs['vaf'], input_af[:, :2, :, :], af_valid_mask)
        loss_haf = 0.5*criterion_reg(outputs['haf'], input_af[:, 2:3, :, :], af_valid_mask)
        pred = torch.sigmoid(outputs['hm']).detach().cpu().numpy()
        target = input_mask.detach().cpu().numpy()
        valid = target != train_loader.dataset.ignore_label
        train_acc = accuracy_score((target[valid] > 0.5).astype(np.int64), (pred[valid] > 0.5).astype(np.int64))
        train_f1 = f1_score((target[valid] > 0.5).astype(np.int64), (pred[valid] > 0.5).astype(np.int64), zero_division=1)

        epoch_loss_seg.append(loss_seg.item())
        for name in hm_channel_names:
            epoch_loss_seg_channels[name].append(loss_seg_channels[name].item())
        epoch_loss_vaf.append(loss_vaf.item())
        epoch_loss_haf.append(loss_haf.item())
        loss = loss_seg + loss_vaf + loss_haf
        epoch_loss.append(loss.item())
        epoch_acc.append(train_acc)
        epoch_f1.append(train_f1)

        loss.backward()
        optimizer.step()
        if b_idx % args.log_schedule == 0:
            global_step = (epoch - 1) * len(train_loader) + b_idx
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f}'.format(
                epoch, (b_idx+1) * args.batch_size, len(train_loader.dataset),
                100. * (b_idx+1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1))
            f_log.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f}\n'.format(
                epoch, (b_idx+1) * args.batch_size, len(train_loader.dataset),
                100. * (b_idx+1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1))
            tb_writer.add_scalar('train_step/loss_seg', loss_seg.item(), global_step)
            for name in hm_channel_names:
                tb_writer.add_scalar(f'train_step/loss_seg_{name}', loss_seg_channels[name].item(), global_step)
            tb_writer.add_scalar('train_step/loss_vaf', loss_vaf.item(), global_step)
            tb_writer.add_scalar('train_step/loss_haf', loss_haf.item(), global_step)
            tb_writer.add_scalar('train_step/loss_total', loss.item(), global_step)
            tb_writer.add_scalar('train_step/accuracy', train_acc, global_step)
            tb_writer.add_scalar('train_step/f1', train_f1, global_step)
            f_log.flush()

    scheduler.step()
    # now that the epoch is completed calculate statistics and store logs
    avg_loss_seg = mean(epoch_loss_seg)
    avg_loss_seg_channels = {name: mean(epoch_loss_seg_channels[name]) for name in hm_channel_names}
    avg_loss_vaf = mean(epoch_loss_vaf)
    avg_loss_haf = mean(epoch_loss_haf)
    avg_loss = mean(epoch_loss)
    avg_acc = mean(epoch_acc)
    avg_f1 = mean(epoch_f1)
    print("\n------------------------ Training metrics ------------------------")
    f_log.write("\n------------------------ Training metrics ------------------------\n")
    print("Average segmentation loss for epoch = {:.2f}".format(avg_loss_seg))
    f_log.write("Average segmentation loss for epoch = {:.2f}\n".format(avg_loss_seg))
    for name in hm_channel_names:
        print("Average {} segmentation loss for epoch = {:.2f}".format(name, avg_loss_seg_channels[name]))
        f_log.write("Average {} segmentation loss for epoch = {:.2f}\n".format(name, avg_loss_seg_channels[name]))
    print("Average VAF loss for epoch = {:.2f}".format(avg_loss_vaf))
    f_log.write("Average VAF loss for epoch = {:.2f}\n".format(avg_loss_vaf))
    print("Average HAF loss for epoch = {:.2f}".format(avg_loss_haf))
    f_log.write("Average HAF loss for epoch = {:.2f}\n".format(avg_loss_haf))
    print("Average loss for epoch = {:.2f}".format(avg_loss))
    f_log.write("Average loss for epoch = {:.2f}\n".format(avg_loss))
    print("Average accuracy for epoch = {:.4f}".format(avg_acc))
    f_log.write("Average accuracy for epoch = {:.4f}\n".format(avg_acc))
    print("Average F1 score for epoch = {:.4f}".format(avg_f1))
    f_log.write("Average F1 score for epoch = {:.4f}\n".format(avg_f1))
    print("------------------------------------------------------------------\n")
    f_log.write("------------------------------------------------------------------\n\n")
    tb_writer.add_scalar('train_epoch/loss_seg', avg_loss_seg, epoch)
    for name in hm_channel_names:
        tb_writer.add_scalar(f'train_epoch/loss_seg_{name}', avg_loss_seg_channels[name], epoch)
    tb_writer.add_scalar('train_epoch/loss_vaf', avg_loss_vaf, epoch)
    tb_writer.add_scalar('train_epoch/loss_haf', avg_loss_haf, epoch)
    tb_writer.add_scalar('train_epoch/loss_total', avg_loss, epoch)
    tb_writer.add_scalar('train_epoch/accuracy', avg_acc, epoch)
    tb_writer.add_scalar('train_epoch/f1', avg_f1, epoch)
    f_log.flush()
    
    return net, avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1

# validation function
def val(net, epoch):
    global best_f1
    epoch_loss_seg, epoch_loss_vaf, epoch_loss_haf, epoch_loss, epoch_acc, epoch_f1 = list(), list(), list(), list(), list(), list()
    epoch_loss_seg_channels = {name: [] for name in hm_channel_names}
    net.eval()

    with torch.no_grad():
        for b_idx, sample in enumerate(val_loader):
            input_img, input_seg, input_mask, input_af = move_sample_to_device(sample, device)

            # do the forward pass
            outputs = net(input_img)[-1]

            # calculate losses and metrics
            af_valid_mask = (input_seg != val_loader.dataset.ignore_label).float()
            loss_seg, loss_seg_channels = compute_segmentation_losses(outputs['hm'], input_mask, val_loader.dataset.ignore_label)
            loss_vaf = 0.5*criterion_reg(outputs['vaf'], input_af[:, :2, :, :], af_valid_mask)
            loss_haf = 0.5*criterion_reg(outputs['haf'], input_af[:, 2:3, :, :], af_valid_mask)
            pred = torch.sigmoid(outputs['hm']).detach().cpu().numpy()
            target = input_mask.detach().cpu().numpy()
            valid = target != val_loader.dataset.ignore_label
            val_acc = accuracy_score((target[valid] > 0.5).astype(np.int64), (pred[valid] > 0.5).astype(np.int64))
            val_f1 = f1_score((target[valid] > 0.5).astype(np.int64), (pred[valid] > 0.5).astype(np.int64), zero_division=1)

            epoch_loss_seg.append(loss_seg.item())
            for name in hm_channel_names:
                epoch_loss_seg_channels[name].append(loss_seg_channels[name].item())
            epoch_loss_vaf.append(loss_vaf.item())
            epoch_loss_haf.append(loss_haf.item())
            loss = loss_seg + loss_vaf + loss_haf
            epoch_loss.append(loss.item())
            epoch_acc.append(val_acc)
            epoch_f1.append(val_f1)

            print('Done with image {} out of {}...'.format(min(args.batch_size*(b_idx+1), len(val_loader.dataset)), len(val_loader.dataset)))

    # now that the epoch is completed calculate statistics and store logs
    avg_loss_seg = mean(epoch_loss_seg)
    avg_loss_seg_channels = {name: mean(epoch_loss_seg_channels[name]) for name in hm_channel_names}
    avg_loss_vaf = mean(epoch_loss_vaf)
    avg_loss_haf = mean(epoch_loss_haf)
    avg_loss = mean(epoch_loss)
    avg_acc = mean(epoch_acc)
    avg_f1 = mean(epoch_f1)
    print("\n------------------------ Validation metrics ------------------------")
    f_log.write("\n------------------------ Validation metrics ------------------------\n")
    print("Average segmentation loss for epoch = {:.2f}".format(avg_loss_seg))
    f_log.write("Average segmentation loss for epoch = {:.2f}\n".format(avg_loss_seg))
    for name in hm_channel_names:
        print("Average {} segmentation loss for epoch = {:.2f}".format(name, avg_loss_seg_channels[name]))
        f_log.write("Average {} segmentation loss for epoch = {:.2f}\n".format(name, avg_loss_seg_channels[name]))
    print("Average VAF loss for epoch = {:.2f}".format(avg_loss_vaf))
    f_log.write("Average VAF loss for epoch = {:.2f}\n".format(avg_loss_vaf))
    print("Average HAF loss for epoch = {:.2f}".format(avg_loss_haf))
    f_log.write("Average HAF loss for epoch = {:.2f}\n".format(avg_loss_haf))
    print("Average loss for epoch = {:.2f}".format(avg_loss))
    f_log.write("Average loss for epoch = {:.2f}\n".format(avg_loss))
    print("Average accuracy for epoch = {:.4f}".format(avg_acc))
    f_log.write("Average accuracy for epoch = {:.4f}\n".format(avg_acc))
    print("Average F1 score for epoch = {:.4f}".format(avg_f1))
    f_log.write("Average F1 score for epoch = {:.4f}\n".format(avg_f1))
    print("--------------------------------------------------------------------\n")
    f_log.write("--------------------------------------------------------------------\n\n")
    tb_writer.add_scalar('val_epoch/loss_seg', avg_loss_seg, epoch)
    for name in hm_channel_names:
        tb_writer.add_scalar(f'val_epoch/loss_seg_{name}', avg_loss_seg_channels[name], epoch)
    tb_writer.add_scalar('val_epoch/loss_vaf', avg_loss_vaf, epoch)
    tb_writer.add_scalar('val_epoch/loss_haf', avg_loss_haf, epoch)
    tb_writer.add_scalar('val_epoch/loss_total', avg_loss, epoch)
    tb_writer.add_scalar('val_epoch/accuracy', avg_acc, epoch)
    tb_writer.add_scalar('val_epoch/f1', avg_f1, epoch)
    f_log.flush()

    # now save the model if it has a better F1 score than the best model seen so forward
    if avg_f1 > best_f1:
        # save the model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'net_' + '%.4d' % (epoch,) + '.pth'))
        best_f1 = avg_f1

    return avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1

if __name__ == "__main__":
    setup()
    heads = {'hm': 3, 'vaf': 2, 'haf': 1}
    model = build_model(args.backbone, heads, device)

    if args.snapshot is not None:
        model = load_snapshot(model, args.snapshot, device)
    else:
        model = model.to(device)
    print(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # BCE(Focal) loss applied to each pixel individually
    model.hm[-1].bias.data.uniform_(-4.595, -4.595) # bias towards negative class
    if args.loss_type == 'focal':
        criterion_1 = FocalLoss(gamma=2.0, alpha=0.25, size_average=True)
    elif args.loss_type == 'bce':
        ## BCE weight
        criterion_1 = torch.nn.BCEWithLogitsLoss()
    elif args.loss_type == 'wbce':
        ## BCE weight
        criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.6, 9.6, 9.6], device=device).view(3, 1, 1))
    criterion_2 = IoULoss()
    criterion_reg = RegL1Loss()

    # set up figures and axes
    fig1, ax1 = plt.subplots()
    plt.grid(True)
    ax1.plot([], 'r', label='Training segmentation loss')
    ax1.plot([], 'g', label='Training VAF loss')
    ax1.plot([], 'b', label='Training HAF loss')
    ax1.plot([], 'k', label='Training total loss')
    ax1.legend()
    train_loss_seg, train_loss_vaf, train_loss_haf, train_loss = list(), list(), list(), list()

    fig2, ax2 = plt.subplots()
    plt.grid(True)
    ax2.plot([], 'r', label='Validation segmentation loss')
    ax2.plot([], 'g', label='Validation VAF loss')
    ax2.plot([], 'b', label='Validation HAF loss')
    ax2.plot([], 'k', label='Validation total loss')
    ax2.legend()
    val_loss_seg, val_loss_vaf, val_loss_haf, val_loss = list(), list(), list(), list()

    fig3, ax3 = plt.subplots()
    plt.grid(True)
    ax3.plot([], 'r', label='Training accuracy')
    ax3.plot([], 'g', label='Validation accuracy')
    ax3.plot([], 'b', label='Training F1 score')
    ax3.plot([], 'k', label='Validation F1 score')
    ax3.legend()
    train_acc, val_acc, train_f1, val_f1 = list(), list(), list(), list()

    # trainval loop
    for i in range(1, args.epochs + 1):
        # training epoch
        model, avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1 = train(model, i)
        train_loss_seg.append(avg_loss_seg)
        train_loss_vaf.append(avg_loss_vaf)
        train_loss_haf.append(avg_loss_haf)
        train_loss.append(avg_loss)
        train_acc.append(avg_acc)
        train_f1.append(avg_f1)
        # plot training loss
        ax1.plot(train_loss_seg, 'r', label='Training segmentation loss')
        ax1.plot(train_loss_vaf, 'g', label='Training VAF loss')
        ax1.plot(train_loss_haf, 'b', label='Training HAF loss')
        ax1.plot(train_loss, 'k', label='Training total loss')
        fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

        # validation epoch
        avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1 = val(model, i)
        val_loss_seg.append(avg_loss_seg)
        val_loss_vaf.append(avg_loss_vaf)
        val_loss_haf.append(avg_loss_haf)
        val_loss.append(avg_loss)
        val_acc.append(avg_acc)
        val_f1.append(avg_f1)
        # plot validation loss
        ax2.plot(val_loss_seg, 'r', label='Validation segmentation loss')
        ax2.plot(val_loss_vaf, 'g', label='Validation VAF loss')
        ax2.plot(val_loss_haf, 'b', label='Validation HAF loss')
        ax2.plot(val_loss, 'k', label='Validation total loss')
        fig2.savefig(os.path.join(args.output_dir, "val_loss.jpg"))

        # plot the train and val metrics
        ax3.plot(train_acc, 'r', label='Train accuracy')
        ax3.plot(val_acc, 'g', label='Validation accuracy')
        ax3.plot(train_f1, 'b', label='Train F1 score')
        ax3.plot(val_f1, 'k', label='Validation F1 score')
        fig3.savefig(os.path.join(args.output_dir, 'trainval_acc_f1.jpg'))

    plt.close('all')
    f_log.close()
