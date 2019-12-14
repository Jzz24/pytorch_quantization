import time
import argparse
import warnings
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.distributed
import torchvision.models as models

cudnn.benchmark = True

from models.resnet_imagenet import *
from utils.preprocess import *
from utils.bar_show import *
import warnings
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='dorefa-net imagenet2012 implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='/imagenet2012_datasets')
parser.add_argument('--log_name', type=str, default='resnet_imagenet_float')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='resnet_float')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=90)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--Wbits', type=int, default=8)
parser.add_argument('--Abits', type=int, default=8)

cfg = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0
TOTAL_TRAIN_PICS = 1271171
TOTAL_EVAL_PICS = 50000

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_dir)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)


def main():

    # nvidia dali dataloader
    train_loader = get_imagenet_iter_dali(type='train', image_dir=cfg.data_dir, batch_size=cfg.train_batch_size,
                                          num_threads=16, crop=224, device_id=0, num_gpus=2)
    eval_loader = get_imagenet_iter_dali(type='val', image_dir=cfg.data_dir, batch_size=cfg.eval_batch_size,
                                          num_threads=8, crop=224, device_id=0, num_gpus=2)

    print('===> Building ResNet..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet18(wbit=cfg.Wbits, abit=cfg.Abits, pretrained=False)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    summary_writer = SummaryWriter(cfg.log_dir)

    if cfg.pretrain:
        ckpt = torch.load(os.path.join(cfg.ckpt_dir, f'checkpoint.t7'))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    for epoch in range(start_epoch, cfg.max_epochs):
        train(epoch, model, train_loader, criterion, optimizer, summary_writer)
        test(epoch, model, eval_loader, criterion, optimizer, summary_writer)
        lr_schedu.step(epoch)
    summary_writer.close()


def train(epoch, model, train_loader, criterion, optimizer, summary_writer):

    print('\nEpoch: %d' % epoch)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, data in enumerate(train_loader):

        #measure data loading time
        data_time.update(time.time() - end)

        inputs = data[0]["data"].cuda(non_blocking=True)
        targets = data[0]["label"].squeeze().long().cuda(non_blocking=True)

        #compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        #measure acc and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        #compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        num_batch_per_epoch = TOTAL_TRAIN_PICS // inputs.size(0) + 1
        progress_bar(batch_idx, num_batch_per_epoch, 'Loss: %.3f | Acc1: %.3f%% Acc5: %.3f%% '
                     % (losses.avg, top1.avg, top5.avg))

        if batch_idx % cfg.log_interval == 0:  #every log_interval mini_batches...
            summary_writer.add_scalar('Loss/train', losses.avg, epoch * num_batch_per_epoch + batch_idx)
            summary_writer.add_scalar('Accuracy/train', top1.avg, epoch * num_batch_per_epoch + batch_idx)
            summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch * num_batch_per_epoch + batch_idx)
            # for tag, value in model.named_parameters():
            #     tag = tag.replace('.', '/')
            #     summary_writer.add_histogram(tag, value.detach(), global_step=epoch * len(train_loader) + batch_idx)
            #     summary_writer.add_histogram(tag + '/grad', value.grad.detach(), global_step=epoch * len(train_loader) + batch_idx)


def test(epoch, model, eval_loader, criterion, optimizer, summary_writer):
    # pass
    global best_acc
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, data in enumerate(eval_loader):

            inputs = data[0]["data"].cuda(non_blocking=True)
            targets = data[0]["label"].squeeze().long().cuda(non_blocking=True)

            #compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            #measure acc and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1,5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            num_batch_per_epoch = TOTAL_EVAL_PICS // inputs.size(0)
            progress_bar(batch_idx, num_batch_per_epoch, 'Loss: %.3f | Acc1: %.3f%% Acc5: %.3f%% '
                         % (losses.avg, top1.avg, top5.avg))

            if batch_idx % cfg.log_interval == 0:  # every log_interval mini_batches...
                summary_writer.add_scalar('Loss/test', losses.avg, epoch * num_batch_per_epoch + batch_idx)
                summary_writer.add_scalar('Accuracy/test', top1.avg, epoch * num_batch_per_epoch + batch_idx)

    acc = top1.avg
    if acc > best_acc:
        print('Saving..')
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(cfg.ckpt_dir, f'checkpoint.t7'))
        best_acc = acc

if __name__ == '__main__':
    main()

