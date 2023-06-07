from torch import nn
import torchvision.models as models

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

class PlainBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(PlainBlock, self).__init__()

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.relu(x)

        return x
    
class PlainNet(nn.Module):
    def __init__(self, layers, num_classes):
        super(PlainNet, self).__init__()        
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self.make_block(64, layers[0])
        self.block2 = self.make_block(128, layers[1], stride=2)
        self.block3 = self.make_block(256, layers[2], stride=2)
        self.block4 = self.make_block(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_block(self, out_channel, num_layers, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel:
            downsample = nn.Sequential(
                conv3x3(out_channel, out_channel, stride),
                nn.BatchNorm2d(out_channel)
            )
        layers = []
        layers.append(PlainBlock(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel

        for _ in range(1, num_layers):
            layers.append(PlainBlock(self.in_channel, out_channel))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

            
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel, stride)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.conv3 = conv1x1(out_channel, out_channel * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride= stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = self.make_block(block, 64, layers[0])
        self.block2 = self.make_block(block, 128, layers[1], stride=2)
        self.block3 = self.make_block(block, 256, layers[2], stride=2)
        self.block4 = self.make_block(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion,  num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_block(self, block, out_channel, num_layers, stride=1):
        ## block : Basic Block or Bottleneck Block
        ## num_layers : The number of layers constituting a block

        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
        ## stride가 1이 아니거나 in_channel이 out_channel * block.expansion이 아닌 경우
        ## shotcut connection의 dimension을 output과 맞추기 위한 projection.
            downsample = nn.Sequential(
                conv1x1(self.in_channel, out_channel * block.expansion, stride),
                nn.BatchNorm2d(out_channel * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion

        for _ in range(1, num_layers):
            layers.append(block(self.in_channel, out_channel))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
def resnet18(num_classes):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
    return model

def resnet34(num_classes):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet50(num_classes):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet101(num_classes):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes)
    return model

def resnet152(num_classes):
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3], num_classes=num_classes)
    return model

def plainnet18(num_classes):
    model = PlainNet(layers=[2, 2, 2, 2], num_classes=num_classes)
    return model

    
if __name__ == "__main__":
    # print(PlainNet(num_classes=100, init_weights=False))
    print(models.resnet34())