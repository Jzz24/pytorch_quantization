import os
import time
import argparse
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True
import torchvision

from models.resnet_cifar import *
from utils.preprocess import *
from utils.bar_show import progress_bar

# Training settings
parser = argparse.ArgumentParser(description='dorefa-net implementation')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='resnet_8w8f_cifar')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='resnet_8w8f_cifar')

parser.add_argument('--cifar', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=250)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--Wbits', type=int, default=8)
parser.add_argument('--Abits', type=int, default=8)

cfg = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_dir)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

def main():
    if cfg.cifar == 10:
        print('training CIFAR-10 !')
        dataset = torchvision.datasets.CIFAR10
    elif cfg.cifar == 100:
        print('training CIFAR-100 !')
        dataset = torchvision.datasets.CIFAR100
    else:
        assert False, 'dataset unknown !'

    print('===> Preparing data ..')
    train_dataset = dataset(root=cfg.data_dir, train=True, download=True,
                          transform=cifar_transform(is_training=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                             num_workers=cfg.num_workers)

    eval_dataset = dataset(root=cfg.data_dir, train=False, download=True,
                         transform=cifar_transform(is_training=False))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                            num_workers=cfg.num_workers)

    print('===> Building ResNet..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ResNet18(wbit=cfg.Wbits,abit=cfg.Abits).to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)
    # optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [90, 150, 200], gamma=0.1)
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


    def train(epoch):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss, correct, total = 0, 0 ,0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            if batch_idx % cfg.log_interval == 0:  #every log_interval mini_batches...
                summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1), epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('Accuracy/train', 100. * correct / total, epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)
                # for tag, value in model.named_parameters():
                #     tag = tag.replace('.', '/')
                #     summary_writer.add_histogram(tag, value.detach(), global_step=epoch * len(train_loader) + batch_idx)
                #     summary_writer.add_histogram(tag + '/grad', value.grad.detach(), global_step=epoch * len(train_loader) + batch_idx)




    def test(epoch):
        # pass
        global best_acc
        model.eval()

        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(eval_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

                if batch_idx % cfg.log_interval == 0:  # every log_interval mini_batches...
                    summary_writer.add_scalar('Loss/test', test_loss / (batch_idx + 1), epoch * len(train_loader) + batch_idx)
                    summary_writer.add_scalar('Accuracy/test', 100. * correct / total, epoch * len(train_loader) + batch_idx)

        acc = 100. * correct / total
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

    for epoch in range(start_epoch, cfg.max_epochs):
        train(epoch)
        test(epoch)
        lr_schedu.step(epoch)
    summary_writer.close()


if __name__ == '__main__':
    main()

