import torch
import torch.nn as nn
from utils.detection_utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        ## Prediction : [grid_size, grid_size, (num_boxes * 5 + num_classes)]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        ## ground-truth box와 pred box 간 IoU 계산.
        pred_box1, pred_box2 = predictions[..., 21:25], predictions[..., 26:30]
        target_box = target[..., 21:25]

        iou_b2 = intersection_over_union(pred_box2, target_box)
        iou_b1 = intersection_over_union(pred_box1, target_box)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0) ## 두 개의 prediction 중 더 높은 iou인 box를 선택. iou_maxes는 가장 높은 값, bestbox는 인덱스.
        exists_box = target[..., 20].unsqueeze(3)  ## 실제 ground-truth의 20번째 원소를 가져오고 unsqueeze. object가 있으면 1이고 없으면 0.

        ## bestbox가 1이면 pred_box2를 선택하고, bestbox가 0이면 pred_box1을 선택.
        box_predictions = exists_box * ((bestbox * pred_box2 + (1 - bestbox) * pred_box1))
        ## 실제 박스 좌표에 exists_box를 곱하여, 객체가 없는 위치의 박스는 0으로 설정
        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        ## bestbox가 1이면 두번째 box의 confidence score를 사용하고, 0이면 첫번째 box의 confidence score를 사용한다.
        pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
        object_loss = self.mse(torch.flatten(exists_box * pred_box), torch.flatten(exists_box * target[..., 20:21]))


        no_object_loss = self.mse(torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1), torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))
        no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1), torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))

        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :20], end_dim=-2,), torch.flatten(exists_box * target[..., :20], end_dim=-2,))

        loss = (self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss)

        return loss