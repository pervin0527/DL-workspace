import torch
from torch import nn

def conv_block(in_channels, out_channels, kernel_size, stride, padding, pooling=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
              nn.BatchNorm2d(out_channels),
              nn.LeakyReLU(0.1, inplace=True)]
    
    if pooling:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)

class YoloV2(nn.Module):
    def __init__(self, num_classes, anchors=None):
        super(YoloV2, self).__init__()
        self.num_classes = num_classes

        if anchors is not None:
            self.anchors = anchors
        else:
            self.anchors = [(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]

        self.stage1 = nn.Sequential(conv_block(3, 32, 3, 1, 1, pooling=True),
                                    conv_block(32, 64, 3, 1, 1, pooling=True),
                                    conv_block(64, 128, 3, 1, 1),
                                    conv_block(128, 64, 1, 1, 0),
                                    conv_block(64, 128, 3, 1, 1, pooling=True),
                                    conv_block(128, 256, 3, 1, 1),
                                    conv_block(256, 128, 1, 1, 0),
                                    conv_block(128, 256, 3, 1, 1, pooling=True),
                                    conv_block(256, 512, 3, 1, 1),
                                    conv_block(512, 256, 1, 1, 0),
                                    conv_block(256, 512, 3, 1, 1),
                                    conv_block(512, 256, 1, 1, 0),
                                    conv_block(256, 512, 3, 1, 1))
        
        self.stage1_max_pool = nn.MaxPool2d(2, 2)

        self.stage2 = nn.Sequential(conv_block(512, 1024, 3, 1, 1),
                                    conv_block(1024, 512, 1, 1, 0),
                                    conv_block(512, 1024, 3, 1, 1),
                                    conv_block(1024, 512, 1, 1, 0),
                                    conv_block(512, 1024, 3, 1, 1),
                                    conv_block(1024, 1024, 3, 1, 1),
                                    conv_block(1024, 1024, 3, 1, 1))

        # Using conv_block for stage2_b and stage3
        self.stage2_conv = conv_block(512, 64, 1, 1, 0)

        self.stage3_conv1 = conv_block(256 + 1024, 1024, 3, 1, 1)
        self.stage3_conv2 = nn.Conv2d(1024, len(self.anchors) * (5 + num_classes), 1, 1, 0, bias=False)

    def forward(self, input):
        # forward pass using the redefined layers
        output = self.stage1(input)
        residual = output

        output_1 = self.stage1_max_pool(output)
        output_1 = self.stage2(output_1)
        output_2 = self.stage2_conv(residual)

        batch_size, num_channel, height, width = output_2.data.size()
        output_2 = output_2.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_2 = output_2.view(batch_size, -1, int(height / 2), int(width / 2))

        output = torch.cat((output_1, output_2), 1)
        output = self.stage3_conv1(output)
        output = self.stage3_conv2(output)

        return output

if __name__ == "__main__":
    from torchsummary import summary

    model = YoloV2(num_classes=20)
    summary(model, input_size=(3, 448, 448), device="cpu")
