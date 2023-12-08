import math
import numpy as np
from torch import nn

def get_model(**kwargs):
    if not "width_multiplier" in kwargs:
        kwargs["width_multiplier"] = 1.0

    model = MobileNetV2(num_classes=kwargs["num_classes"], width_multiple=kwargs["width_multiplier"])
    
    return model

def make_divisible(x, divisible_by=8):
    ## 네트워크의 너비(채널 수)를 divisible_by의 배수로 맞춘다.
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def conv_bn_relu6(in_channels, out_channels, stride):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU6(inplace=True))

    return block

def conv_1x1_bn_relu6(in_channels, out_channels):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU6(inplace=True))
    
    return block


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.residual_connection = stride == 1 and in_channels == out_channels

        ## expand_ratio가 1이면 expansion을 하지 않는다.
        if expand_ratio == 1:
            self.block = nn.Sequential(
                ## Depth-Wise Convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                ## Point-Wise Convolution
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                ## Point-Wise Convolution for Expansion
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                ## Depth-Wise Convolution
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),

                ## Point-Wise Convolution for Bottleneck
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        if self.residual_connection:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, width_multiple=1.):
        super(MobileNetV2, self).__init__()
        first_channel, last_channel = 32, 1280

        ## t, c, n, s : expansion_factor, out_channels, num_block, stride
        inverted_residual_setting = [[1, 16, 1, 1],
                                     [6, 24, 2, 2],
                                     [6, 32, 3, 2],
                                     [6, 64, 4, 2],
                                     [6, 96, 3, 1],
                                     [6, 160, 3, 2],
                                     [6, 320, 1, 1]]
        self.last_channel = make_divisible(last_channel * width_multiple) if width_multiple > 1.0 else last_channel
        
        self.feature_extractor = [conv_bn_relu6(in_channels=3, out_channels=first_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_multiple) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.feature_extractor.append(InvertedResidualBlock(first_channel, output_channel, s, expand_ratio=t))
                else:
                    self.feature_extractor.append(InvertedResidualBlock(first_channel, output_channel, 1, expand_ratio=t))
                first_channel = output_channel

        self.feature_extractor.append(conv_1x1_bn_relu6(first_channel, self.last_channel))
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        self.classifier = nn.Linear(self.last_channel, num_classes)

        self.initialize_weights()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()