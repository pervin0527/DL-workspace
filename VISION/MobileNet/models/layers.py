import torch

from torch import nn

class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthWiseSeparableConv2d, self).__init__()
        
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU(inplace=True))
        
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        return x