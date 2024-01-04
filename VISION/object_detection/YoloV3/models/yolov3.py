from torch import nn

from modles.darknet53 import build_darknet53
from models.layers import conv_bn_relu_layer, conv_bn_relu_block, conv_output_layer, DetectionLayer

class YoloV3(nn.Moudle):
    def __init__(self, anchors, img_size, num_classes):
        final_out_channel = 3 * (4 + 1 + num_classes) ## grid cell마다 3개의 bounding box를 예측한다.

        self.darknet53 = build_darknet53()
        self.conv_block1 = conv_bn_relu_layer(in_channels=1024, out_channels=512)
        self.conv_final1 = conv_output_layer(in_channels=512, out_channels=final_out_channel)
        self.head1 = DetectionLayer(anchors[3], img_size, num_classes)