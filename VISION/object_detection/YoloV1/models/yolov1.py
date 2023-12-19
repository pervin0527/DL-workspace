import torch
from torch import nn

def conv_bn_relu(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                         nn.BatchNorm2d(out_num),
                         nn.LeakyReLU())

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_bn_relu(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_bn_relu(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual

        return out
    
class Darknet53(nn.Module):
    def __init__(self, num_classes):
        super(Darknet53, self).__init__()
        self.num_classes = num_classes

        self.conv1 = conv_bn_relu(3, 32)
        self.conv2 = conv_bn_relu(32, 64, stride=2)
        self.residual_block1 = nn.Sequential(*[ResidualBlock(64) for _ in range(1)])
        self.conv3 = conv_bn_relu(64, 128, stride=2)
        self.residual_block2 = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])
        self.conv4 = conv_bn_relu(128, 256, stride=2)
        self.residual_block3 = nn.Sequential(*[ResidualBlock(256) for _ in range(8)])
        self.conv5 = conv_bn_relu(256, 512, stride=2)
        self.residual_block4 = nn.Sequential(*[ResidualBlock(512) for _ in range(8)])
        self.conv6 = conv_bn_relu(512, 1024, stride=2)
        self.residual_block5 = nn.Sequential(*[ResidualBlock(1024) for _ in range(4)])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        # out = out.view(-1, 1024)
        # out = self.fc(out)

        out = out.view(out.size(0), -1)

        return out
    
    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        for key in list(state_dict.keys()):
            if "fc.weight" in key or "fc.bias" in key:
                del state_dict[key]

        self.load_state_dict(state_dict, strict=False)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)

        return x

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.S = kwargs["grid_size"]
        self.B = kwargs["num_boxes"]
        self.C = kwargs["num_classes"]

        self.darknet = Darknet53(kwargs["num_classes"])
        self.darknet.load_weights(kwargs["pretrained"])
        
        # self.darknet = self.build_darknet(in_channels)
        
        self.head = self.build_head(kwargs["grid_size"], kwargs["num_boxes"], kwargs["num_classes"])

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        
        return x

    def build_darknet(self, in_channels):
        layers = [CNNBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
                  nn.MaxPool2d(kernel_size=2, stride=2),

                  CNNBlock(64, 192, kernel_size=3, stride=1, padding=1),
                  nn.MaxPool2d(kernel_size=2, stride=2),
      
                  CNNBlock(192, 128, kernel_size=1, stride=1, padding=0),
                  CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
                  CNNBlock(256, 256, kernel_size=1, stride=1, padding=0),
                  CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
                  nn.MaxPool2d(kernel_size=2, stride=2)]

        for _ in range(4):
            layers.append(CNNBlock(512, 256, kernel_size=1, stride=1, padding=0))
            layers.append(CNNBlock(256, 512, kernel_size=3, stride=1, padding=1))

        layers += [CNNBlock(512, 512, kernel_size=1, stride=1, padding=0),
                   CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
                   nn.MaxPool2d(kernel_size=2, stride=2)]

        for _ in range(2):
            layers.append(CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0))
            layers.append(CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1))

        layers += [CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
                   CNNBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
                   CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
                   CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1)]

        return nn.Sequential(*layers)

    def build_head(self, grid_size, num_boxes, num_classes):
        S, B, C = grid_size, num_boxes, num_classes

        return nn.Sequential(nn.Linear(1024, 4096),
                             # nn.Linear(1024 * S * S, 4096),
                             nn.LeakyReLU(0.1),
                             nn.Linear(4096, S * S * (C + B * 5)))

    
if __name__ == "__main__":
    from torchsummary import summary

    # model = Darknet53(num_classes=10)
    # model.load_weights("/home/pervinco/Models/darknet53_best.pth.tar")
    # summary(model, input_size=(3, 448, 448), device="cpu")

    model = Yolov1(in_channels=3, grid_size=7, num_boxes=2, num_classes=20)
    summary(model, input_size=(3, 448, 448), device="cpu")
