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

        ## no_object, coordinate coefficient(논문 - loss function에 기재된 lambda값들.)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        ## predictions : [batch_size, S * S * (B * 5 + C)]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5) ## [batch_size, S, S, (B * 5 + C)]

        ## Cell마다 예측된 두 개의 pred_box와 ground_truth box간 IoU 계산.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        ## 계산된 IoU를 쌓아서 하나의 텐서로 만든다.
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
 
        ## IoU가 가장 높은 하나의 box를 선택한다. iou_maxes는 가장 높은 IoU값, bestbox는 index로 0 또는 1의 값을 갖는다.
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  ## 논문상 Iobj_ij에 해당하는 것으로, object가 해당 cell에 포함되어 있는지, 있으면 1 없으면 0이다.

        ## Bounding Box Coordinates.
        ## bestbox가 0이면 bestbox * predictions[..., 26:30]가 0이되고, 1이면 (1 - bestbox) * predictions[..., 21:25]이 0이 된다.
        ## IoU가 더 높은 바운딩 박스의 예측값을 추출.
        box_predictions = exists_box * ((bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]))

        box_targets = exists_box * target[..., 21:25] ## ground-truth의 box를 가져온다.

        ## 너비와 높이 가져오기.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        ## Object loss
        ## IoU가 가장 높은 box에 대한 confidence score를 선별한다.
        pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])

        object_loss = self.mse(torch.flatten(exists_box * pred_box), torch.flatten(exists_box * target[..., 20:21]))

        ## No object loss
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1), torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))
        no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1), torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))


        ## Class Score loss
        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :20], end_dim=-2,), torch.flatten(exists_box * target[..., :20], end_dim=-2,))

        loss = (self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss)

        return loss
