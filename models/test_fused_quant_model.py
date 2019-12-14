from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

from resnet_cifar import *
from model_utils.quant_dorefa import QuanConv as Conv


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
    def forward(self, x):
        return x

def fuse(conv, bn):

    global fuse_layer_idx   ## control first layer fuse
    fuse_layer_idx += 1
    # *******************conv params********************
    w = conv.weight

    # ********************BN params*********************
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight
    beta = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * gamma + beta

    if fuse_layer_idx == 1:
        print ('fuse first layer')
        fused_conv = nn.Conv2d(conv.in_channels,
                               conv.out_channels,
                               conv.kernel_size,
                               conv.stride,
                               conv.padding,
                               groups=conv.groups,
                               bias=True)
    else:
        fused_conv = Conv(in_channels=conv.in_channels,
                          out_channels=conv.out_channels,
                          kernel_size=conv.kernel_size,
                          nbit_w=32,
                          nbit_a=args.Abits,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def fuse_modules(m):
    children = list(m.named_children())
    conv, conv_name= None, None
    for layer_idx, (name, child) in enumerate(children):
        if isinstance(child, nn.BatchNorm2d):
            fused_conv = fuse(conv, child)
            print ('===> fusing')
            m._modules[conv_name] = fused_conv
            m._modules[name] = DummyModule()
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            fuse_modules(child)

def model_convert():

    model = ResNet18(wbit=32,abit=args.Abits)  ## modify model config to w=32
    model_state_dict = torch.load(args.baseline_model_dir + 'checkpoint.t7')['model_state_dict']

    ## fix up nn.DataParallel module load to cpu model's bug
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    # **********************convert to W qunatized model*************************
    conv_lay_idx = 0   ## usually do not quantify the first and last layer
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_lay_idx += 1
            if conv_lay_idx>=2:
                m.weight.data = dorefa_w(m.weight.data, nbit_w=args.Wbits)
            elif isinstance(m, nn.Linear):
                m.weight.data = dorefa_w(m.weight.data, nbit_w=8)
                # m.weight.data = m.weight.data
            else:
                pass

    torch.save(model, args.baseline_model_dir + 'quan_model.pth')  #save entire model
    torch.save(model.state_dict(), args.baseline_model_dir + 'quan_model_para.pth') #save model state_dict


    # ********************** convert to bn_fold W quantized model *************************
    model.eval()
    fuse_modules(model)
    torch.save(model, args.baseline_model_dir + 'quan_bn_merged_model.pth')
    torch.save(model.state_dict, args.baseline_model_dir + 'quan_bn_merged_model_para.pth')

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    test_time = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = Variable(data.cuda()), Variable(target.cuda())

            start_time = time.time()
            output = model(data)
            end_time = time.time()

            test_loss += criterion(output, target).data.item()
            test_time += end_time - start_time
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        acc = 100. * float(correct) / len(testloader.dataset)
        test_loss /= len(testloader.dataset)
        print('\nmodel: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Running Time: {:.2f} s'.format(
            test_loss * 100, correct, len(testloader.dataset), acc, test_time))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='../data',help='dataset path')
    parser.add_argument('--baseline_model_dir', type=str, default='../ckpt/resnet_8w8f_cifar/')
    parser.add_argument('--eval_batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--Wbits', type=int, default=8)
    parser.add_argument('--Abits', type=int, default=8)
    args = parser.parse_args()

    print('==> Options:', args)
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False,
                                             num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()

    fuse_layer_idx = 0

    model_convert()

    # load entire model
    quan_model = torch.load(args.baseline_model_dir + 'quan_model.pth')
    quan_bn_merged_model = torch.load(args.baseline_model_dir + 'quan_bn_merged_model.pth')

    if not args.cpu:
        quan_model.cuda()
        quan_bn_merged_model.cuda()
    test(quan_model)
    test(quan_bn_merged_model)

