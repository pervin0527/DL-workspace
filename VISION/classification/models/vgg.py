import torch
from torch import nn


def get_model(**kwargs):
    model = VGG19(num_classes=kwargs["num_classes"])
    
    return model


def conv_bn_relu(in_channels, out_channels, num_layers):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    
    for _ in range(num_layers):
        layers.extend([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)])
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    return nn.Sequential(*layers)


class VGG19(nn.Module):
    def __init__(self, num_classes, init_weights=False):
        super(VGG19, self).__init__()
        ## input : [3, 224, 224]

        ## Feature Extractor
        self.block1 = conv_bn_relu(in_channels=3, out_channels=64, num_layers=1) ## [3, 224, 224] -> [64, 112, 112]
        self.block2 = conv_bn_relu(in_channels=64, out_channels=128, num_layers=1) ## [64, 112, 112] -> [128, 56, 56]
        self.block3 = conv_bn_relu(in_channels=128, out_channels=256, num_layers=3) ## [128, 56, 56] -> [256, 28, 28]
        self.block4 = conv_bn_relu(in_channels=256, out_channels=512, num_layers=3) ## [256, 28, 28] -> [512, 14, 14]
        self.block5 = conv_bn_relu(in_channels=512, out_channels=512, num_layers=4) ## [512, 14, 14] -> [512, 7, 7]

        ## Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) ## [512, 7, 7]

        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), ## 25088 -> 4096
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5),

                                        nn.Linear(4096, 4096), ## 4096 -> 4096
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(4096, num_classes))
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) ## [512, 7, 7] to 25088

        x = self.classifier(x)

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