import os
import time
import numpy
import argparse
import warnings
from datetime import datetime

import torch
import torchvision.datasets as datasets
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
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
parser.add_argument('--log_name', type=str, default='resnet_imagenet_4w4f')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='resnet_4w4f')


parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=90)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--Wbits', type=int, default=4)
parser.add_argument('--Abits', type=int, default=4)

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:2345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


cfg = parser.parse_args()


best_acc = 0  # best test accuracy
start_epoch = 0

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_dir)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

def main():

    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(cfg.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)

    print('===> Building ResNet..')

    model = resnet18(wbit=cfg.Wbits, abit=cfg.Abits, pretrained=False)
    # model = models.__dict__[resnet18(pretrained=False)]

    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.train_batch_size = int(cfg.train_batch_size / ngpus_per_node)
            cfg.workers = int((cfg.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()


    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().cuda(cfg.gpu)
    summary_writer = SummaryWriter(cfg.log_dir)

    # Data loading code
    traindir = os.path.join(cfg.data_dir, 'train')
    valdir = os.path.join(cfg.data_dir, 'val')

    train_dataset = datasets.ImageFolder(traindir, imgnet_transform(is_training=True))
    val_dataset = datasets.ImageFolder(valdir,imgnet_transform(is_training=False))

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=(train_sampler is None), num_workers=cfg.num_workers, pin_memory=True, sampler=train_sampler)
    eval_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)



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
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
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
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #measure data loading time
        data_time.update(time.time() - end)

        if cfg.gpu is not None:
            inputs = inputs.cuda(cfg.gpu, non_blocking=True)
        targets = targets.cuda(cfg.gpu, non_blocking=True)
        # inputs, targets = inputs.to('cuda'), targets.to('cuda')

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

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc1: %.3f%% Acc5: %.3f%% '
                     % (losses.avg, top1.avg, top5.avg))

        if batch_idx % cfg.log_interval == 0:  #every log_interval mini_batches...
            summary_writer.add_scalar('Loss/train', losses.avg, epoch * len(train_loader) + batch_idx)
            summary_writer.add_scalar('Accuracy/train', top1.avg, epoch * len(train_loader) + batch_idx)
            summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)
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
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            if cfg.gpu is not None:
                inputs = inputs.cuda(cfg.gpu, non_blocking=True)
            targets = targets.cuda(cfg.gpu, non_blocking=True)

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

            progress_bar(batch_idx, len(eval_loader), 'Loss: %.3f | Acc1: %.3f%% Acc5: %.3f%% '
                         % (losses.avg, top1.avg, top5.avg))

            if batch_idx % cfg.log_interval == 0:  # every log_interval mini_batches...
                summary_writer.add_scalar('Loss/test', losses.avg, epoch * len(eval_loader) + batch_idx)
                summary_writer.add_scalar('Accuracy/test', top1.avg, epoch * len(eval_loader) + batch_idx)

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