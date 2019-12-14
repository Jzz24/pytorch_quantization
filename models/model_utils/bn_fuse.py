import torch
import torch.nn as nn
import time
import sys
import numpy as np
import torchvision
import torch.nn.functional as F


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        # print("Dummy, Dummy.")
        return x

def fuse(conv, bn):
    # *******************conv参数********************
    w = conv.weight

    # ********************BN参数*********************
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

    # fused_conv = Conv(in_channels=conv.in_channels,
    #                      out_channels=conv.out_channels,
    #                      kernel_size=conv.kernel_size,
    #                      nbit_w=32,
    #                      nbit_a=8,
    #                      stride=conv.stride,
    #                      padding=conv.padding,
    #                      groups=conv.groups,
    #                      bias=True)
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def fuse_module(m):
    children = list(m.named_children())
    c = None
    cn = None

    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_module(child)


def test_net(m):

    p = torch.randn([1, 3, 224, 224])
    import time
    s = time.time()
    o_output = m(p)
    print("Original time: ", time.time() - s)

    fuse_module(m)
    # print(m)

    s = time.time()
    f_output = m(p)
    print("Fused time: ", time.time() - s)

    print("Max abs diff: ", (o_output - f_output).abs().max().item())
    assert(o_output.argmax() == f_output.argmax())
    # print(o_output[0][0].item(), f_output[0][0].item())
    print("MSE diff: ", nn.MSELoss()(o_output, f_output).item())

def fuse_model(m):
    p = torch.rand([1, 3, 32, 32])
    s = time.time()
    o_output = m(p)
    print("Original time: ", time.time() - s)

    fuse_module(m)

    s = time.time()
    f_output = m(p)
    print("Fused time: ", time.time() - s)
    return m


def test():
    print("============================")
    print("Module level test: ")
    m = torchvision.models.resnet18(True)
    m.eval()
    test_net(m)
    # fuse_model(m)


if __name__ == '__main__':
    test()