import math
from torch import nn
from models.layers import DepthWiseSeparableConv2d, InvertedResidual, make_divisible, conv_bn, conv_1x1_bn

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),

                                   DepthWiseSeparableConv2d(32, 64, 1),
                                   DepthWiseSeparableConv2d(64, 128, 2),
                                   DepthWiseSeparableConv2d(128, 128, 1),
                                   DepthWiseSeparableConv2d(128, 256, 2),
                                   DepthWiseSeparableConv2d(256, 256, 1),
                                   DepthWiseSeparableConv2d(256, 512, 2),   
                                   DepthWiseSeparableConv2d(512, 512, 1),
                                   DepthWiseSeparableConv2d(512, 512, 1),
                                   DepthWiseSeparableConv2d(512, 512, 1),
                                   DepthWiseSeparableConv2d(512, 512, 1),
                                   DepthWiseSeparableConv2d(512, 512, 1),   
                                   DepthWiseSeparableConv2d(512, 1024, 2),
                                   DepthWiseSeparableConv2d(1024, 1024, 1),
                                   nn.AvgPool2d(7))
        self.fc = nn.Linear(1024, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x
    

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Linear(self.last_channel, num_classes)

        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
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