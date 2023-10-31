from torch import nn
from models.layers import DepthWiseSeparableConv2d

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(MobileNetV1, self).__init__()
        self.init_weights = init_weights

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

        if self.init_weights:
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