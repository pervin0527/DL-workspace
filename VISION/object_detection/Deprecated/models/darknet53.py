from torch import nn
from models.layers import conv_bn_relu_layer, residual_block

def build_darknet53():
    modules = nn.ModuleDict()

    modules['conv_1'] = conv_bn_relu_layer(in_channels=3, out_channels=32, kernel_size=3, requires_grad=False)
    modules['conv_2'] = conv_bn_relu_layer(in_channels=32, out_channels=64, kernel_size=3, stride=2, requires_grad=False)

    modules["residual_1_1"] = residual_block(in_channels=64)
    
    modules["conv_3"] = conv_bn_relu_layer(in_channels=64, out_channels=128, kernel_size=3, stride=2, requires_grad=False)
    modules["residual_2_1"] = residual_block(in_channels=128)
    modules["residual_2_2"] = residual_block(in_channels=128)

    modules["conv_4"] = conv_bn_relu_layer(in_channels=128, out_channels=256, kernel_size=3, stride=2, requires_grad=False)
    modules["residual_3_1"] = residual_block(in_channels=256)
    modules["residual_3_2"] = residual_block(in_channels=256)
    modules["residual_3_3"] = residual_block(in_channels=256)
    modules["residual_3_4"] = residual_block(in_channels=256)
    modules["residual_3_5"] = residual_block(in_channels=256)
    modules["residual_3_6"] = residual_block(in_channels=256)
    modules["residual_3_7"] = residual_block(in_channels=256)
    modules["residual_3_8"] = residual_block(in_channels=256)

    modules["conv_5"] = conv_bn_relu_layer(in_channels=256, out_channels=512, kernel_size=3, stride=2, requires_grad=False)
    modules['residual_4_1'] = residual_block(in_channels=512)
    modules['residual_4_2'] = residual_block(in_channels=512)
    modules['residual_4_3'] = residual_block(in_channels=512)
    modules['residual_4_4'] = residual_block(in_channels=512)
    modules['residual_4_5'] = residual_block(in_channels=512)
    modules['residual_4_6'] = residual_block(in_channels=512)
    modules['residual_4_7'] = residual_block(in_channels=512)
    modules['residual_4_8'] = residual_block(in_channels=512)

    modules["conv_6"] = conv_bn_relu_layer(in_channels=512, out_channels=1024, kernel_size=3, stride=2, requires_grad=False)
    modules['residual_5_1'] = residual_block(in_channels=1024)
    modules['residual_5_2'] = residual_block(in_channels=1024)
    modules['residual_5_3'] = residual_block(in_channels=1024)
    modules['residual_5_4'] = residual_block(in_channels=1024)

    return modules