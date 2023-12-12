from torch import nn

VALID_MODELS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")


def get_model(**kwargs):
    model = ResNet.build_from_name(kwargs["model_name"], num_classes=kwargs["num_classes"])

    return model


def resnet_params(model_name):
    params_dict = {"resnet18" : (BasicBlock, [2, 2, 2, 2]),
                   "renset34" : (BasicBlock, [3, 4, 6, 3]),
                   "resnet50" : (Bottleneck, [3, 4, 6, 3]),
                   "resnet101" : (Bottleneck, [3, 4, 23, 3]),
                   "resnet152" : (Bottleneck, [3, 8, 36, 3])}
    
    return params_dict[model_name]


def get_model_params(model_name):
    block, layers = resnet_params(model_name)

    return block, layers


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        ## 1x1 conv for Bottleneck.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        ## 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        ## 1x1 conv
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
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
        self.in_channels = 64

        ## block1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## block2 ~
        self.block1 = self.make_block(block, 64, layers[0])
        self.block2 = self.make_block(block, 128, layers[1], stride=2)
        self.block3 = self.make_block(block, 256, layers[2], stride=2)
        self.block4 = self.make_block(block, 512, layers[3], stride=2)

        ## Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion,  num_classes)

        ## init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def make_block(self, block, out_channels, num_layers, stride=1):
        """
        block : Basic Block or Bottleneck Block
        num_layers : Number of layers that make up a block
        """
        downsample = None
        ## Make residual connection possible by matching channel values between identity and output.
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels * block.expansion))
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_layers):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    @ classmethod
    def build_from_name(cls, model_name, num_classes):
        cls.check_model_name_is_valid(model_name)

        block, layers = get_model_params(model_name)
        model = cls(block, layers, num_classes)

        return model

    @classmethod
    def check_model_name_is_valid(cls, model_name):
        if model_name not in VALID_MODELS:
            raise ValueError("model_name should be one of : " + ", ".join(VALID_MODELS))
        

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
        