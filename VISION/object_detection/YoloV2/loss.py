import math
import torch
from torch import nn

class YoloLoss(nn.modules.loss._Loss):
    # The loss I borrow from LightNet repo.
    def __init__(self, num_classes, anchors, reduction=32, coord_scale=1.0, noobject_scale=1.0, object_scale=5.0, class_scale=1.0, thresh=0.6):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

    def forward(self, output, target):
        ## output : [-1, 125, 13, 13]
        ## 125 = 5[num_boxes per grid_cell] * (coords[x, y, w, h] + confidence_score + num_classes[20]) 
        batch_size = output.data.size(0) ## batch_size
        height = output.data.size(2) ## 13
        width = output.data.size(3) ## 13

        ##  (t_x, t_y, t_w, t_h), conf_score, classification_scores를 가져온다. -> 즉 anchor를 얼마나 이동시키고 조절할지에 대한 Offset이다.
        ## [batch_size, 125, 13, 13] -> [batch_size, 5, 25, 169] = [batch_size, num_boxes, (num_classes + coords + conf_score), grid_cells]
        output = output.view(batch_size, self.num_anchors, -1, height * width)

        coord = torch.zeros_like(output[:, :, :4, :]) ## [batch_size, 5, 4, 169] + zeros
        
        ## cx, cy, w, h를 가져오는데, 중심좌표에는 sigmoid를 적용한다.
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()
        coord[:, :, 2:4, :] = output[:, :, 2:4, :]

        ## confidence score.
        conf = output[:, :, 4, :].sigmoid()

        ## classification scores
        ## [batch_size, 125, 20, 169] -> [batch_size * 5, 20, 169] -> [batch_size * 5, 169, 20] -> [(batch_size * 5 * 169), 20]
        cls = output[:, :, 5:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes, height * width).transpose(1, 2).contiguous().view(-1,self.num_classes)

        ## Create prediction boxes
        pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4) ## [bathc_size * 5 * 13 * 13, 4]
        
        ## grid cell의 x, y 좌표 생성.
        lin_x = torch.arange(0, width).repeat(height, 1).view(height * width)
        lin_y = torch.arange(0, height).repeat(width, 1).t().contiguous().view(height * width)

        ## 정의된 앵커 박스의 너비와 높이를 가져와서 num_anchors x 1 크기의 텐서로 변환
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)

        if torch.cuda.is_available():
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        ## 모델의 출력인 coord에서 바운딩 박스의 중심 좌표 (x, y)와 크기 (width, height)를 실제 좌표로 변환.
        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1) ## anchor box의 중심점을 모델이 예측한 offset만큼 이동 시킨다.
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)

        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1) ## anchor box의 height, width에 예측한 offset을 반영해 조절한다.
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        pred_boxes = pred_boxes.cpu()

        ## 여기까지가 t_x, t_y, t_w, t_h, t_o를 활용해 b_x, h_y, b_h, b_w를 만드는 과정.

        ## b_x, b_y, b_w, b_h와 ground_truth [batch_size, [xmin, ymin, xmax, ymax, label]]를 사용해서 loss를 계산한다.
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target, height, width)
        coord_mask = coord_mask.expand_as(tcoord)
        tcls = tcls[cls_mask].view(-1).long()
        cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)

        if torch.cuda.is_available():
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            tcls = tcls.cuda()
            cls_mask = cls_mask.cuda()

        conf_mask = conf_mask.sqrt()
        cls = cls[cls_mask].view(-1, self.num_classes)

        # Compute losses
        mse = nn.MSELoss(reduction="sum")
        ce = nn.CrossEntropyLoss(reduction="sum")
        self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask) / batch_size
        self.loss_conf = mse(conf * conf_mask, tconf * conf_mask) / batch_size
        self.loss_cls = self.class_scale * 2 * ce(cls, tcls) / batch_size
        self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls

        return self.loss_tot, self.loss_coord, self.loss_conf, self.loss_cls

    def build_targets(self, pred_boxes, ground_truth, height, width):
        batch_size = len(ground_truth)
        """
        - conf_mask : confidence score 마스크. 모든 값을 1로 초기화하고 noobject_scale을 곱한다.(객체가 존재하지 않는다고 가정)
        - coord_mask : 바운딩 박스 좌표 마스크.
        - cls_mask : class scores 마스크.
        - tcoord : 실제 바운딩 박스의 좌표 마스크.
        - tconf : 실제 객체의 존재 여부를 나타내는 마스크.
        - tcls: 타겟 클래스는 실제 객체의 클래스 정보를 저장.
        """

        conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * self.noobject_scale ## [batch_size, num_anchors, 169]        
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False) ## [batch_size, num_anchors, 1, 169]
        cls_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).bool() ## [batch_size, num_anchors, 169]
        tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False) ## [batch_size, num_anchors, 4, 169]
        tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False) ## [batch_size, num_anchors, 169]
        tcls = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False) ## [batch_size, nnum_anchors, 169]

        ## 0 ~ 31 batch에 포함된 각각의 ground-truth에 접근.
        for b in range(batch_size):
            if len(ground_truth[b]) == 0: ## 현재 이미지에 ground truth가 없으면 object가 없는 것이므로 넘어간다.
                continue

            cur_pred_boxes = pred_boxes[b * (self.num_anchors * height * width) : (b + 1) * (self.num_anchors * height * width)] ## 현재 이미지에 대한 예측된 바운딩 박스를 선택.
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1) ## 0으로 채워진 텐서와 원래의 앵커 텐서를 연결. [0] * 5와 [(w, h)] * 5

            ## loss 계산을 위해 ground_truth의 xmin, ymin, xmax, ymax를 변환한다.
            gt = torch.zeros(len(ground_truth[b]), 4) ## [object의 수, 4]
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction ## (xmin + xmax) / 32
                gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction ## (ymin + ymax) / 32
                gt[i, 2] = anno[2] / self.reduction ## width / 32
                gt[i, 3] = anno[3] / self.reduction ## height / 32

            iou_gt_pred = bbox_ious(gt, cur_pred_boxes) ## Ground truth 바운딩 박스와 예측된 바운딩 박스 간의 IOU(Intersection over Union)를 계산.
            
            ##  IOU가 설정된 임계값(threshold)보다 박스의 수를 측정. 예측된 바운딩 박스가 최소 한 개의 ground truth 박스와 임계값 이상의 IOU를 가지는 경우를 찾는다.
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0 ## object가 있는 곳을 0으로 표기.

            gt_wh = gt.clone()
            gt_wh[:, :2] = 0 ## 바운딩 박스의 너비와 높이만을 사용하기 위해 중심 좌표를 0으로 설정
            iou_gt_anchors = bbox_ious(gt_wh, anchors) ## gt 박스와 앵커 간의 IOU를 계산
            _, best_anchors = iou_gt_anchors.max(1) ## 각 ground truth 바운딩 박스에 대해 가장 높은 IOU 값을 가진 앵커를 찾는다.

            for i, anno in enumerate(ground_truth[b]): ## 각 ground truth 바운딩 박스에 대해
                ## gt의 중심 좌표를 그리드 셀 크기에 맞게 조정.
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))

                ## 각 ground truth 바운딩 박스에 가장 잘 맞는 앵커 인덱스를 선택.
                best_n = best_anchors[i]
                
                ## 선택된 앵커에 대한 IOU 값을 계산. 이 값은 예측된 바운딩 박스와 실제 ground truth 바운딩 박스 간의 일치 정도를 나타낸다.
                iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]

                ## 해당 좌표에 객체가 존재함을 나타내는 마스크를 설정
                coord_mask[b][best_n][0][gj * width + gi] = 1

                ## 클래스 마스크를 설정
                cls_mask[b][best_n][gj * width + gi] = 1

                ## 체 존재에 대한 신뢰도 마스크를 설정
                conf_mask[b][best_n][gj * width + gi] = self.object_scale

                ##  ground truth 바운딩 박스의 실제 좌표와 해당 그리드 셀, 앵커 박스의 좌표 사이의 차이를 계산하고 저장.
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])

                ## 계산된 IOU 값을 신뢰도 타겟으로 설정.
                tconf[b][best_n][gj * width + gi] = iou

                ## 각 ground truth 바운딩 박스의 클래스 인덱스를 설정.
                tcls[b][best_n][gj * width + gi] = int(anno[4])

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls


def bbox_ious(boxes1, boxes2):
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions
