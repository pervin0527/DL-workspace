import torch.nn as nn

class PlainNet(nn.Module):
    def __init__(self, num_classes, init_weights):
        super(PlainNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self.make_block(num_conv=3, in_channels=56, out_channels=56)
        self.block2 = self.make_block(num_conv=4, in_channels=128, out_channels=56, first_stride=2)
        self.block3 = self.make_block(num_conv=6, in_channels=256, out_channels=56, first_stride=2)
        self.block4 = self.make_block(num_conv=3, in_channels=512, out_channels=56, first_stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(7*1*1, num_classes)

        if init_weights:
            self.initialize_weights()
        

    def forward(self, x):
        x = self.conv1(x) ## 112, 112
        x = self.pool1(x) ## 56, 56

        x = self.block1(x) ## 56, 56
        x = self.block2(x) ## 28, 28
        x = self.block3(x) ## 14, 14
        x = self.block4(x) ## 7, 7

        x = self.avgpool(x)
        x = self.fc(x)

        return x
    
    def make_block(self, num_conv, in_channels, out_channels, first_stride=0):
        block = []

        for i in range(num_conv):
            if first_stride != 0 and i == 0:
                conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            else:
                conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

            bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            relu = nn.ReLU(inplace=True)

            conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            relu = nn.ReLU(inplace=True)

            block += [conv1, bn1, relu, conv2, bn2, relu]

        return nn.Sequential(*block)
    
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

    
if __name__ == "__main__":
    print(PlainNet(num_classes=100, init_weights=False))