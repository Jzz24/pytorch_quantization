import torch
import torch.nn as nn
import numpy as np

# from .model_utils.quant_dorefa import QuanConv as Conv
# from .model_utils.quant_dorefa import *
# from .model_utils.bn_fuse import fuse_module

from model_utils.quant_dorefa import QuanConv as Conv
from model_utils.quant_dorefa import *
from model_utils.bn_fuse import fuse_module

import torch.nn.functional as F


def conv3x3(in_planes, out_planes, wbit, abit, stride=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, nbit_w=wbit, nbit_a=abit)


def conv1x1(in_planes, out_planes, wbit, abit, stride=1):
    """1x1 convolution"""
    return Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, nbit_w=wbit, nbit_a=abit)


def linear(in_featrues, out_features, wbit, abit):
    return Linear_Q(in_featrues, out_features, nbit_w=wbit, nbit_a=abit)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, wbit, abit, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, wbit=wbit, abit=abit, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, wbit=wbit, abit=abit, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes, wbit=wbit, abit=abit, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, wbit, abit, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes, wbit=wbit, abit=abit, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, wbit=wbit, abit=abit, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion*planes,wbit=wbit, abit=abit, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion*planes,wbit=wbit,abit=abit,stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, wbit, abit, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # do not quntize the first layer
        # self.conv1 = Conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], wbit=wbit, abit=abit, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], wbit=wbit, abit=abit, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], wbit=wbit, abit=abit, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], wbit=wbit, abit=abit, stride=2)
        self.linear = linear(512*block.expansion, num_classes, wbit=8, abit=abit)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def _make_layer(self, block, planes, num_blocks, wbit, abit, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, wbit, abit, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(wbit, abit):
    return ResNet(BasicBlock, [2,2,2,2], wbit=wbit, abit=abit)

def ResNet34(wbit, abit):
    return ResNet(BasicBlock, [3,4,6,3], wbit=wbit, abit=abit)

def ResNet50(wbit, abit):
    return ResNet(Bottleneck, [3,4,6,3], wbit=wbit, abit=abit)

def ResNet101(wbit, abit):
    return ResNet(Bottleneck, [3,4,23,3], wbit=wbit, abit=abit)

def ResNet152(wbit, abit):
    return ResNet(Bottleneck, [3,8,36,3], wbit=wbit, abit=abit)


def test():
    net = ResNet18(wbit=32,abit=32)
    print (net)
    fuse_module(net)
    print (net)
    y = net(torch.ones(1,3,32,32))
    print(y)

if __name__ == '__main__':
    test()

