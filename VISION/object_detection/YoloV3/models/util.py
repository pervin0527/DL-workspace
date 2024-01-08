import torch
from data.util import xywh2xyxy

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """Returns the IoU of two bounding boxes."""
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, device):
    """
    pred_boxes : [num_batches, num_anchors, grid_size, grid_size, 4]
    pred_cls : [num_batches, num_anchors, grid_size, grid_size, self.num_classes]
    target : [objects_in_images,  class_ids, x, y, w, h]

    example
    tensor([[ 0.0000,  6.0000,  0.5403,  0.7812,  0.5194,  0.4375],
            [ 0.0000,  6.0000,  0.6097,  0.1635,  0.4194,  0.3229],
            [ 0.0000,  6.0000,  0.4375,  0.5135,  0.6028,  0.6604],
            [ 1.0000, 14.0000,  0.7044,  0.2716,  0.5297,  0.5432],
            [ 1.0000, 12.0000,  0.3617,  0.5375,  0.7234,  0.9250]])
    """
    nB = pred_boxes.size(0) ## batch_size
    nA = pred_boxes.size(1) ## num_anchors
    nG = pred_boxes.size(2) ## grid_size
    nC = pred_cls.size(-1)  ## num_classes

    # Output tensors
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool, device=device)     ## [batch_size, num_anchors, grid_size, grid_size]
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool, device=device)    ## [batch_size, num_anchors, grid_size, grid_size]
    class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)  ## [batch_size, num_anchors, grid_size, grid_size]
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)  ## [batch_size, num_anchors, grid_size, grid_size]

    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float, device=device)
    tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float, device=device)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG ## bx, by, bw, bh를 grid scale에 맞게 변환.
    gxy = target_boxes[:, :2] ## grid scale의 bx, by
    gwh = target_boxes[:, 2:] ## bw, bh

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors]) ## anchor와 bw, bh간 iou를 계산.
    _, best_ious_idx = ious.max(0) ## 가장 높은 iou를 계산.

    # Separate target values
    b, target_labels = target[:, :2].long().t() ## object가 포함된 이미지 idx, class_label
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # Set masks
    obj_mask[b, best_ious_idx, gj, gi] = 1 ## obj_mask의 best_iou_idx를 True로 설정.
    noobj_mask[b, best_ious_idx, gj, gi] = 0 ## noobj_mask의 best_iou_idx를 False로 설정.

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0 ## anchor iou가 ignore_thres보다 높으면 객체가 존재하는 것이므로 noobj_mask를 False로 설정.

    """
    ground-truth 박스를 이용해 offset을 계산한다.
    """
    # Coordinates
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor() ## i, j번째 grid cell 내부에서 중심점이 몇 인지 계산 [0, 1]
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16) ## 박스의 width, height
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    # One-hot encoding of label
    tcls[b, best_ious_idx, gj, gi, target_labels] = 1 ## i, j번째 grid cell에 있는 객체의 클래스

    # Compute label correctness and iou at best anchor
    class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == target_labels).float() ## 올바른 클래스를 예측했는지 마스크에 체크. 맞췄으면 1, 틀렸으면 0
    iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], target_boxes, x1y1x2y2=False) ## pred box와 target box 간 Iou.

    tconf = obj_mask.float()

    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def non_max_suppression(prediction, img_height, img_width, conf_thres, nms_thres):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4], img_height=img_height, img_width=img_width)
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output