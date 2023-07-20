import math
import torch

def clip_boxes_to_image(boxes, size):
    ## 이미지 영역을 벗어나지 않는 anchor들만 선별한다.
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)

    return clipped_boxes.reshape(boxes.shape)

def remove_small_boxes(boxes, min_size):
    ## 작은 크기의 box는 제거.
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = keep.nonzero().squeeze(1)

    return keep

def nms(boxes, scores, iou_threshold):
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)

def batched_nms(boxes, scores, idxs, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

def box_area(boxes):
    ## box 넓이 계산. (xmax - xmin) * (ymax - ymin)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    ## IoU 계산
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def encode_boxes(reference_boxes, proposals, weights):
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1

    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

    return targets

class BoxCoder(object):
    ## bounding box regression을 위한 coordinate encode/decode.
    def __init__(self, weights, bbox_xform_clip=math.log(1000./16)):
        self.weights = weights ## regression weights.
        self.bbox_xfrom_clip = bbox_xform_clip ## 변환 범위에 대한 clip : log(max_size / 16)

    def encode(self, reference_boxes, proposals):
        ## regression parameter를 계산하기 위함.
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)
    
    def encode_single(self, reference_boxes, proposals):
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets
    
    def decode(self, rel_codes, boxes):
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        
        pred_bboxes = self.decode_single(rel_codes.reshape(box_sum, -1), concat_boxes)

        return pred_bboxes.reshape(box_sum, -1, 4)
    
    def decode_single(self, rel_codes, boxes):
        ## 원본 box들과 encoding된 relative box의 offset들로부터 decoding된 box들을 가져온다.
        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]  ## anchor width
        heights = boxes[:, 3] - boxes[:, 1] ## anchor height
        ctr_x = boxes[:, 0] + 0.5 * widths  ## anchor center x
        ctr_y = boxes[:, 1] + 0.5 * heights ## anchor centery y

        wx, wy, ww, wh = self.weights  ## default is 1
        dx = rel_codes[:, 0::4] / wx   ## predicated anchors center x regression parameters
        dy = rel_codes[:, 1::4] / wy   ## predicated anchors center y regression parameters
        dw = rel_codes[:, 2::4] / ww   ## predicated anchors width regression parameters
        dh = rel_codes[:, 3::4] / wh   ## predicated anchors height regression parameters

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w ## xmin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h ## ymin
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w ## xmax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h ## ymax
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        
        return pred_boxes
    
def set_low_quality_matches_(matches, all_matches, match_quality_matrix):
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

    gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])

    pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
    matches[pre_inds_to_update] = all_matches[pre_inds_to_update]
    

class Matcher(object):
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        ## hight_threshold : 이 값보다 큰 경우 positive label
        ## low_threshold : 이 값보다 낮은 경우 negative label
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2 ## lower < IoU < higher일 경우 -2를 할당.
        
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        ## anchor box와 ground-truth box 사이에 IoU를 계산하고, anchor의 index를 저장한다.
        ## iou < low_threshold: -1
        ## iou > high_threshold: 1
        ## low_threshold<=iou<high_threshold: -2

        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")

        ## match_quality_matrix : M (gt) x N (predicted)
        ## Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        ## Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches
    
class BalancedPositiveNegativeSampler(object):
    ## batch image 당 sample anchor(논문상 256개)를 sampling한다.
    ## batch_size_per_image : batch단위에 포함되는 이미지의 수.
    ## positive_fraction : 256개 중에서 positive의 비율

    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)

            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
    

def permute_and_flatten(layer, N, A, C, H, W):
    ## 순서를 조정하고 shape을 변경
    ## [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    
    return layer

    
def concat_box_prediction_layers(box_cls, box_regression):
    ## box classification & bbox regerssion parameter 정렬 및 shape 조정. return: [N, -1, C]

    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, H, W = box_cls_per_level.shape ## [batch_size, anchors_num_per_position * classes_num, height, width], class_num is equal 2
        Ax4 = box_regression_per_level.shape[1] ## [batch_size, anchors_num_per_position * 4, height, width]
        A = Ax4 // 4
        C = AxC // A

        ## [N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        ## [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)

    return box_cls, box_regression

def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    
    if size_average:
        return loss.mean()
    
    return loss.sum()
