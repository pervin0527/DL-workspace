import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

def save_feats_mean(x):
    b, c, h, w = x.shape
    if h == 256:
        with torch.no_grad():
            x = x.detach().cpu().numpy()
            x = np.transpose(x[0], (1, 2, 0))
            x = np.mean(x, axis=-1)
            x = x/np.max(x)
            x = x * 255.0
            x = x.astype(np.uint8)
            x = cv2.applyColorMap(x, cv2.COLORMAP_JET)
            x = np.array(x, dtype=np.uint8)

            return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU(),
                                  nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_c))
        
        self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
                                      nn.BatchNorm2d(out_c))

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.r1 = ResidualBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.r1(inputs)
        p = self.pool(x)

        return x, p

class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, dim, num_layers=2):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(out_c),
                                   nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.tblock = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.conv2 = nn.Sequential(nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_c),
                                   nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        b, c, h, w = x.shape
        x = x.reshape((b, c, h*w))
        x = self.tblock(x)
        x = x.reshape((b, c, h, w))
        x = self.conv2(x)

        return x


class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, dilation=1),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU())

        self.c2 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=3, dilation=3),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU())

        self.c3 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=6, dilation=6),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU())

        self.c4 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, padding=9, dilation=9),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU())

        self.c5 = nn.Sequential(nn.Conv2d(out_c*4, out_c, kernel_size=1, padding=0),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU())

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.c5(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0]+in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)

        return x


class TResUnet(nn.Module):
    def __init__(self, backbone, num_layers=2):
        super().__init__()

        """ Backbone """
        if backbone.lower() == "resnet50":
            # backbone = resnet50()
            backbone = models.resnet50(weights="IMAGENET1K_V2")
        elif backbone.lower() == "resnet101":
            # backbone = resnet101()
            backbone = models.resnet101(weights="IMAGENET1K_V2")
        elif backbone.lower() == "resnet152":
            # backbone = resnet152()
            backbone = models.resnet152(weights="IMAGENET1K_V2")

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        """ Bridge blocks """
        self.b1 = Bottleneck(2048, 512, 64, num_layers=2)
        self.b2 = DilatedConv(2048, 512)

        """ Decoder """
        self.d1 = DecoderBlock([1024, 1024], 512)
        self.d2 = DecoderBlock([512, 512], 256)
        self.d3 = DecoderBlock([256, 256], 128)
        self.d4 = DecoderBlock([128, 64], 64)
        self.d5 = DecoderBlock([64, 3], 32)

        self.output = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x, heatmap=None):
        s0 = x
        s1 = self.layer0(s0)
        s2 = self.layer1(s1)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)

        b1 = self.b1(s5)
        b2 = self.b2(s5)
        b3 = torch.cat([b1, b2], axis=1)

        d1 = self.d1(b3, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        d5 = self.d5(d4, s0)

        y = self.output(d5)

        if heatmap != None:
            hmap = save_feats_mean(d5)
            return hmap, y
        else:
            return y