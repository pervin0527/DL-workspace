import torch
from torch import nn
from models.util import build_targets

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

def conv_bn_relu_block(in_channels, out_channels):
    expanded_channels = out_channels * 2
    modules = nn.Sequential(conv_bn_relu_layer(in_channels, out_channels, kernel_size=1, padding=0),
                            conv_bn_relu_layer(out_channels, expanded_channels, kernel_size=3),
                            conv_bn_relu_layer(expanded_channels, out_channels, kernel_size=1, padding=0),
                            conv_bn_relu_layer(out_channels, expanded_channels, kernel_size=3),
                            conv_bn_relu_layer(expanded_channels, out_channels, kernel_size=1, padding=0))
    return modules


def residual_block(in_channels):
    middle_channel = in_channels // 2
    block = nn.Sequential(conv_bn_relu_layer(in_channels, middle_channel, kernel_size=1, padding=0, requires_grad=False),
                          conv_bn_relu_layer(middle_channel, in_channels, kernel_size=3, requires_grad=False))
    
    return block


def conv_upsample(in_channels, out_channels, scale_factor):
    modules = nn.Sequential(conv_bn_relu_layer(in_channels, out_channels, kernel_size=1, padding=0),
                            nn.Upsample(scale_factor=scale_factor, mode="nearest"))
    
    return modules


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
        self.no_obj_scale = 100
        self.metrics = {}

    def forward(self, x, targets):
        device = torch.device('cuda' if x.is_cuda else 'cpu')

        num_batches = x.size(0)
        grid_size = x.size(2)

        ## x : [batch_size, grid_size, grid_size, num_anchors * (5 + num_classes)]
        ## x.view : [batch_size, num_anchors, 5 + num_classes, grid_size, grid_size]
        ## permute : [batch_size, num_anchors, grid_size, grid_size, 5 + num_classes]
        prediction = (x.view(num_batches, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous())

        ## 출력값 정리
        ## prediction : [batch_size, num_anchors, grid_size, grid_size, 5 + num_classes]
        cx = torch.sigmoid(prediction[..., 0]) ## tx
        cy = torch.sigmoid(prediction[..., 1]) ## ty
        w = prediction[..., 2] ## tw
        h = prediction[..., 3] ## th
        pred_conf = torch.sigmoid(prediction[..., 4]) ## confidence score(objectness) [num_batches, num_anchors, grid_size, grid_size, 1]
        pred_cls = torch.sigmoid(prediction[..., 5:]) ## class score [num_batches, num_anchors, grid_size, grid_size, self.num_classes]

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

        ## 모델이 예측한 offset들을 조정된 앵커 박스에 반영한다. tx, ty, tw, th --> bx, by, bw, bh
        pred_boxes = torch.zeros_like(prediction[..., :4], device=device) ## [batch_size, num_anchors, grid_size, grid_size, 4]
        pred_boxes[..., 0] = cx + grid_x
        pred_boxes[..., 1] = cy + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        ## pred_boxes : [num_batches, num_anchors, grid_size, grid_size, 4]
        ## pred_conf : [num_batches, num_anchors, grid_size, grid_size, 1]
        ## pred_cls : [num_batches, num_anchors, grid_size, grid_size, self.num_classes]
        pred = (pred_boxes.view(num_batches, -1, 4) * stride, pred_conf.view(num_batches, -1, 1), pred_cls.view(num_batches, -1, self.num_classes))
        output = torch.cat(pred, -1)

        if targets is None:
            return output, 0
        
        iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = build_targets(pred_boxes=pred_boxes,
                                                                                                   pred_cls=pred_cls,
                                                                                                   target=targets,
                                                                                                   anchors=scaled_anchors,
                                                                                                   ignore_thres=self.ignore_threshold,
                                                                                                   device=device)
        
        # Loss: Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse(cx[obj_mask], tx[obj_mask])
        loss_y = self.mse(cy[obj_mask], ty[obj_mask])
        loss_w = self.mse(w[obj_mask], tw[obj_mask])
        loss_h = self.mse(h[obj_mask], th[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h
        loss_conf_obj = self.bce(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj
        loss_cls = self.bce(pred_cls[obj_mask], tcls[obj_mask])
        loss_layer = loss_bbox + loss_conf + loss_cls

        # Metrics
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_no_obj = pred_conf[no_obj_mask].mean()
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        # Write loss and metrics
        self.metrics = {
            "loss_x": loss_x.detach().cpu().item(),
            "loss_y": loss_y.detach().cpu().item(),
            "loss_w": loss_w.detach().cpu().item(),
            "loss_h": loss_h.detach().cpu().item(),
            "loss_bbox": loss_bbox.detach().cpu().item(),
            "loss_conf": loss_conf.detach().cpu().item(),
            "loss_cls": loss_cls.detach().cpu().item(),
            "loss_layer": loss_layer.detach().cpu().item(),
            "cls_acc": cls_acc.detach().cpu().item(),
            "conf_obj": conf_obj.detach().cpu().item(),
            "conf_no_obj": conf_no_obj.detach().cpu().item(),
            "precision": precision.detach().cpu().item(),
            "recall50": recall50.detach().cpu().item(),
            "recall75": recall75.detach().cpu().item()
        }

        return output, loss_layer