import torch
from torch import nn

def conv_bn_relu_layer(in_channels, out_channels, kernel_size, stride=1, padding=1, requires_grad=True):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9, eps=1e-5)
    relu = nn.LeakyReLU(negative_slope=0.1)

    if not requires_grad:
        for param in conv.parameters():
            param.requires_grad_(False)
        for param in bn.parameters():
            param.requires_grad_(False)

    layer = nn.Sequential(conv, bn, relu)

    return layer


def conv_bn_relu_block(in_channels):
    middle_channel = in_channels // 2
    block = nn.Sequential(conv_bn_relu_layer(in_channels, middle_channel, kernel_size=1, padding=0, requires_grad=False),
                          conv_bn_relu_layer(middle_channel, in_channels, kernel_size=3, requires_grad=False))
    
    return block


def conv_output_layer(in_channels, out_channels):
    layer = nn.Sequential(conv_bn_relu_layer(in_channels, in_channels * 2, kernel_size=3),
                          nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
    
    return layer


class DetectionLayer(nn.Module):
    def __init__(self, anchors, img_size, num_classes):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ignore_threshold = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}

    def forward(self, x, targets):
        device = torch.device('cuda' if x.is_cuda else 'cpu')

        num_batches = x.size(0)
        grid_size = x.size(2)

        prediction = (x.view(num_batches, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous())

        ## 출력값 정리
        cx = torch.sigmoid(prediction[..., 0]) ## tx
        cy = torch.sigmoid(prediction[..., 1]) ## ty
        w = prediction[..., 2] ## tw
        h = prediction[..., 3] ## th
        pred_conf = torch.sigmoid(prediction[..., 4]) ## confidence score(objectness)
        pred_cls = torch.sigmoid(prediction[..., 5:]) ## class score

        ## grid 정의.
        stride = self.img_size / grid_size ## 전체 downscale factor
        grid_x = torch.arange(grid_size, dtype=torch.float, device=device)
        grid_x = grid_x.repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, dtype=torch.float, device=device)
        grid_y = grid_y.repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])

        ## anchor의 너비와 높이를 grid 범위로 조정한다.
        scaled_anchors = torch.as_tensor([(pw / stride, ph / stride) for pw, ph in self.anchors], dtype=torch.float, device=device)
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        ## 모델이 예측한 offset들을 조정된 앵커 박스에 반영한다. -> bx, by, bw, bh
        pred_boxes = torch.zeros_like(prediction[..., :4], device=device)
        pred_boxes[..., 0] = cx + grid_x
        pred_boxes[..., 1] = cy + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h