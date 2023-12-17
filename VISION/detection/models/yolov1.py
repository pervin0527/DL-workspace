import torch
from torch import nn

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
        self.darknet = self.build_darknet(in_channels)
        self.head = self.build_head(**kwargs)

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

        return nn.Sequential(nn.Linear(1024 * S * S, 496),
                             nn.LeakyReLU(0.1),
                             nn.Linear(496, S * S * (C + B * 5)))
    
if __name__ == "__main__":
    from torchsummary import summary
    model = model = Yolov1(in_channels=3, grid_size=7, num_boxes=2, num_classes=20)
    summary(model, input_size=(3, 448, 448), device="cpu")