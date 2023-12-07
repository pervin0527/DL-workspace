from torch import nn

def get_model(**kwargs):
    model = MobileNetV1(num_classes=kwargs["num_classes"])
    
    return model


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    
    layers = nn.Sequential(*layers)

    return layers


class DepthWiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthWiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       nn.ReLU())
        
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU())
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(self, num_classes, init_weights=False):
        super(MobileNetV1, self).__init__()
        self.feature_extractor = nn.Sequential(conv_bn_relu(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1), ## [32, 112, 112]
                                               
                                               DepthWiseSeparableConv2d(in_channels=32, out_channels=64, stride=1), ## [64, 112, 112]
                                               
                                               DepthWiseSeparableConv2d(64, 128, 2), ## [128, 56, 56]
                                               DepthWiseSeparableConv2d(128, 128, 1), ## [128, 56, 56]

                                               DepthWiseSeparableConv2d(128, 256, 2), ## [256, 28, 28]
                                               DepthWiseSeparableConv2d(256, 256, 1), ## [256, 28, 28]
                                               
                                               DepthWiseSeparableConv2d(256, 512, 2), ## [512, 14, 14]
                                               DepthWiseSeparableConv2d(512, 512, 1), ## [512, 14, 14]
                                               DepthWiseSeparableConv2d(512, 512, 1), ## [512, 14, 14]
                                               DepthWiseSeparableConv2d(512, 512, 1), ## [512, 14, 14]
                                               DepthWiseSeparableConv2d(512, 512, 1), ## [512, 14, 14]
                                               DepthWiseSeparableConv2d(512, 512, 1), ## [512, 14, 14]
                                               
                                               DepthWiseSeparableConv2d(512, 1024, 2), ## [1024, 7, 7]
                                               DepthWiseSeparableConv2d(1024, 1024, 1), ## [1024, 7, 7]
         )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(1024, num_classes)

        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)

        return x
    
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