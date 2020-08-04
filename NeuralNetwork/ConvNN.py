import torch.nn as nn
import torch.nn.functional as F

class Conv_BN_2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding):
        super(Conv_BN_2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(x))

class Conv_BN_ReLU_2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding):
        super(Conv_BN_ReLU_2d, self).__init__()
        self.conv_bn = Conv_BN_2d(in_planes, out_planes, kernel_size, padding=padding)

    def forward(self, x):
        return F.relu(self.conv_bn(x), inplace=True)
           
    
def Conv1x1_BN(in_planes, out_planes):
    return Conv_BN_2d(in_planes, out_planes, 1, 0)

def Conv1x1_BN_ReLU(in_planes, out_planes):
    return Conv_BN_ReLU_2d(in_planes, out_planes, 1, 0)

def Conv3x3_BN(in_planes, out_planes):
    return Conv_BN_2d(in_planes, out_planes, 3, 1)

def Conv3x3_BN_ReLU(in_planes, out_planes):
    return Conv_BN_ReLU_2d(in_planes, out_planes, 3, 1)
