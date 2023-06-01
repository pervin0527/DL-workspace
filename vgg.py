import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, model_name, num_classes, init_weights):
        super(VGG, self).__init__()
        self.cfg = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], 
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], 
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        self.features = self.feature_extractor(self.cfg[model_name])
        # self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
    
    def initialize_weights(self):
        for m in self.modules(): ## self.modules : 정의된 layer들을 담고 있다.
            if isinstance(m, nn.Conv2d): ## 해당 항목이 torch.nn.Conv2d와 동일한 클래스인가?
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') ## he normal init
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) ## bias를 0으로 초기화한다.
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def feature_extractor(self, block_info, batch_norm=False):
        layers = []
        in_channels = 3

        for v in block_info:
            if v == 'M':# max pooling
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)