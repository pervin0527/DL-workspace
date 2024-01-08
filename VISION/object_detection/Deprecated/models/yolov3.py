import torch
import numpy as np

from torch import nn

from models.darknet53 import build_darknet53
from models.layers import conv_bn_relu_block, conv_bn_relu_block, conv_output_layer, conv_upsample, DetectionLayer

class YoloV3(nn.Module):
    def __init__(self, anchors, img_size, num_classes):
        super(YoloV3, self).__init__()
        final_out_channel = 3 * (4 + 1 + num_classes) ## grid cell마다 3개의 bounding box를 예측한다.

        self.darknet53 = build_darknet53()
        self.conv_block3 = conv_bn_relu_block(in_channels=1024, out_channels=512)
        self.conv_final3 = conv_output_layer(in_channels=512, out_channels=final_out_channel)
        self.head3 = DetectionLayer(anchors[2], img_size, num_classes)

        self.upsample2 = conv_upsample(in_channels=512, out_channels=256, scale_factor=2)
        self.conv_block2 = conv_bn_relu_block(in_channels=768, out_channels=256)
        self.conv_final2 = conv_output_layer(in_channels=256, out_channels=final_out_channel)
        self.head2 = DetectionLayer(anchors[1], img_size, num_classes)

        self.upsample1 = conv_upsample(in_channels=256, out_channels=128, scale_factor=2)
        self.conv_block1 = conv_bn_relu_block(in_channels=384, out_channels=128)
        self.conv_final1 = conv_output_layer(in_channels=128, out_channels=final_out_channel)
        self.head1 = DetectionLayer(anchors[0], img_size, num_classes)

        self.heads = [self.head1, self.head2, self.head3]

    def forward(self, x, targets=None):
        loss = 0
        residual_output = {}

        # Darknet-53 forward
        with torch.no_grad():
            for key, module in self.darknet53.items():
                module_type = key.split('_')[0]

                if module_type == 'conv':
                    x = module(x)
                elif module_type == 'residual':
                    out = module(x)
                    x += out
                    if key == 'residual_3_8' or key == 'residual_4_8' or key == 'residual_5_4':
                        residual_output[key] = x

        # Yolov3 layer forward
        conv_block3 = self.conv_block3(residual_output['residual_5_4'])
        scale3 = self.conv_final3(conv_block3)
        yolo_output3, layer_loss = self.head3(scale3, targets)
        loss += layer_loss

        scale2 = self.upsample2(conv_block3)
        scale2 = torch.cat((scale2, residual_output['residual_4_8']), dim=1)
        conv_block2 = self.conv_block2(scale2)
        scale2 = self.conv_final2(conv_block2)
        yolo_output2, layer_loss = self.head2(scale2, targets)
        loss += layer_loss

        scale1 = self.upsample1(conv_block2)
        scale1 = torch.cat((scale1, residual_output['residual_3_8']), dim=1)
        conv_block1 = self.conv_block1(scale1)
        scale1 = self.conv_final1(conv_block1)
        yolo_output1, layer_loss = self.head1(scale1, targets)
        loss += layer_loss

        yolo_outputs = [yolo_output1, yolo_output2, yolo_output3]
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()

        return yolo_outputs if targets is None else (loss, yolo_outputs)
    

    # Load original weights file
    def load_darknet_weights(self, weights_path: str):
        # Open the weights file
        with open(weights_path, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values (0~2: version, 3~4: seen)
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        # Load Darknet-53 weights
        for key, module in self.darknet53.items():
            module_type = key.split('_')[0]

            if module_type == 'conv':
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            elif module_type == 'residual':
                for i in range(2):
                    ptr = self.load_bn_weights(module[i][1], weights, ptr)
                    ptr = self.load_conv_weights(module[i][0], weights, ptr)

        # Load YOLOv3 weights
        if weights_path.find('yolov3.weights') != -1:
            for module in self.conv_block3:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final3[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final3[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final3[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final3[1], weights, ptr)

            ptr = self.load_bn_weights(self.upsample2[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.upsample2[0][0], weights, ptr)

            for module in self.conv_block2:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final2[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final2[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final2[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final2[1], weights, ptr)

            ptr = self.load_bn_weights(self.upsample1[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.upsample1[0][0], weights, ptr)

            for module in self.conv_block1:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final1[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final1[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final1[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final1[1], weights, ptr)


    # Load BN bias, weights, running mean and running variance
    def load_bn_weights(self, bn_layer, weights, ptr: int):
        num_bn_biases = bn_layer.bias.numel()

        # Bias
        bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.bias)
        bn_layer.bias.data.copy_(bn_biases)
        ptr += num_bn_biases
        # Weight
        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_weights)
        ptr += num_bn_biases
        # Running Mean
        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_running_mean)
        ptr += num_bn_biases
        # Running Var
        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_running_var)
        ptr += num_bn_biases

        return ptr


    # Load convolution weights
    def load_conv_weights(self, conv_layer, weights, ptr: int):
        num_weights = conv_layer.weight.numel()

        conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
        conv_weights = conv_weights.view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_weights)
        ptr += num_weights

        return ptr


    # Load convolution bias
    def load_conv_bias(self, conv_layer, weights, ptr: int):
        num_biases = conv_layer.bias.numel()

        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases]).view_as(conv_layer.bias)
        conv_layer.bias.data.copy_(conv_biases)
        ptr += num_biases

        return ptr