import torch
import torch.nn.functional as F

from utils.detection_utils import generate_all_anchors, xywh2xxyy, box_transform_inv, box_ious, xxyy2xywh, box_transform


def build_target(output, gt_data, H, W, train_params):
    delta_pred_batch = output[0]    ## [batch_size, height * width * num_boxes, 4] pred of delta. σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred_batch = output[1]     ## [batch_size, height * width * num_boxes, 1] pred if confidence score.
    class_score_batch = output[2]   ## [batch_size, height * width * num_boxes, num_classes] pred of class scores.

    gt_boxes_batch = gt_data[0]     ## [batch_size, num_of_object, 4] ground truth boxes(normalized values 0 ~ 1)
    gt_classes_batch = gt_data[1]   ## [batch_1ize, num_of_object] ground truth classes.
    num_boxes_batch = gt_data[2]    ## [batch_size, 1] num of objects.

    bsize = delta_pred_batch.size(0)
    num_anchors = 5  # hard code for now

    ## initial the output tensor(결과값을 저장할 빈 텐서.)
    iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * train_params["noobject_scale"]

    box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    """
    모든 anchor의 x, y, w, h는 grid의 width, height에 의해 정규화된 상태이다. 즉, 앵커 박스의 크기가 전체 이미지의 크기가 아닌 해당 그리드 셀의 크기에 따라 정의되었다.
    모델의 예측은 0에서 1 사이의 값으로 정규화되어 있다. 따라서 모델이 예측하는 객체의 위치, 크기 등이 그리드 셀의 상대적인 크기와 위치를 기반으로 계산되어야 한다.
    note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13.
    this is very crucial because the predict output is normalized to 0~1, which is also normalized by the grid width and height.
    """
    ## 모든 앵커를 그리드 셀 범위로 변환시키고 그에 대한 cx, cy, w, h를 만든다.
    anchors = torch.FloatTensor(train_params["anchors"]) ## [[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892], [9.47112, 4.84053], [11.2364, 10.0071]]
    all_grid_xywh = generate_all_anchors(anchors, H, W) ## shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh) ## delta_pred_batch와 같은 타입, 장치를 할당받는 텐서를 만드는 것.
    all_anchors_xywh = all_grid_xywh.clone() ## 그리드 셀에 대한 앵커 박스들을 복제.
    all_anchors_xywh[:, 0:2] += 0.5 ## 각 앵커 박스의 x, y 중심 좌표를 0.5씩 증가시킨다.
    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh) ## 좌표변환 x,y,w,h -> xmin, xmax, ymin, ymax. 각 그리드셀에 대한 이미지 범위의 앵커 박스.

    ## batch를 구성하는 batch_size개의 원소에 접근.
    for b in range(bsize):
        num_obj = num_boxes_batch[b].item() ## batch의 i번째 데이터가 가진 object의 수.
        delta_pred = delta_pred_batch[b]    ## i번째 데이터의 x, y, w, h
        gt_boxes = gt_boxes_batch[b][:num_obj, :] ## i번째 데이터의 ground truth box.
        gt_classes = gt_classes_batch[b][:num_obj] ## i번째 데이터의 classes.

        ## rescale ground truth boxes.
        gt_boxes[:, 0::2] *= W ## 0번 idx부터 시작해서 2단계씩 이동. 각 박스의 x, w에 W를 곱한다.
        gt_boxes[:, 1::2] *= H

        ## STEP 1: process IoU target.
        # apply delta_pred to pre-defined anchors
        all_anchors_xywh = all_anchors_xywh.view(-1, 4) ## (H * W * num_anchors, 4)
        
        ## ## 그리드 셀에 대한 앵커 박스를 모델의 예측만큼 이동, 크기 조절.
        box_pred = box_transform_inv(all_grid_xywh, delta_pred) ## [batch_size, h*w*num_anchors, 4]
        ## 좌표 변환. 각 그리드셀에 대한 이미지 범위의 예측 박스.
        box_pred = xywh2xxyy(box_pred) ## [batch_size, h*w*num_anchors, 4]

        ## gt box와 IoU가 가장 높은 pred +  anchor box를 찾는다.
        ious = box_ious(box_pred, gt_boxes) ## [845, num_obj]
        ious = ious.view(-1, num_anchors, num_obj) ## [h*w, 5, 1]
        max_iou, _ = torch.max(ious, dim=-1, keepdim=True) # (H * W, num_anchors, 1)

        """ Hard-Negative Mining. """
        iou_thresh_filter = max_iou.view(-1) > train_params["thres"] ## thres 이상의 iou는 True 낮은 iou는 False인 Boolean 텐서.
        n_pos = torch.nonzero(iou_thresh_filter).numel() ## 0이 아닌(True) 원소들을 남겨두고 numel은 이들의 수를 카운트한다.

        if n_pos > 0:
            iou_mask[b][max_iou >= train_params["thres"]] = 0
        """ 이 과정을 통해 전체 영역내에서 배경과 객체에 대한 구분이 가능해진다. """

        ## STEP 2: process box target and class target.
        ## calculate overlaps between anchors and gt boxes
        """ gt_box와 가장 적합한 앵커 박스를 선택하고 이들 간 offset을 구하여 target, mask에 기록한다."""
        overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj) ## 모든 xxyy 앵커 박스와 gt 박스간 IOU를 계산한다.
        gt_boxes_xywh = xxyy2xywh(gt_boxes)

        # iterate over all objects
        for t in range(gt_boxes.size(0)):
            # compute the center of each gt box to determine which cell it falls on assign it to a specific anchor by choosing max IoU.
            gt_box_xywh = gt_boxes_xywh[t] ## t번째 gt box의 좌표
            gt_class = gt_classes[t] ## t번째 gt class
            cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2]) ## t번째 gt box의 중심에 floor를 적용해 grid cell의 i, j를 구한다.
            cell_idx = cell_idx_y * W + cell_idx_x ## 2차원 그리드 셀 인덱스를 1차원 인덱스로 변환. 169 중 i번째.
            cell_idx = cell_idx.long()

            ## update box_target, box_mask
            overlaps_in_cell = overlaps[cell_idx, :, t] ##  현재 그리드 셀에 대한 모든 앵커 박스와 t번째 정답 박스 간의 IOU를 가져온다.
            argmax_anchor_idx = torch.argmax(overlaps_in_cell) ## 가장 높은 IOU를 보인 앵커 박스의 인덱스를 선택.

            assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0) ## 선택된 앵커 박스의 정보를 가져온다.
            gt_box = gt_box_xywh.unsqueeze(0)
            target_t = box_transform(assigned_grid, gt_box) ## 선택된 앵커 박스와 gt 박스간의 offset을 계산한다. σ(t_x), σ(t_y), exp(t_w), exp(t_h)

            box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0) ## 계산된 변환을 박스 타겟 텐서에 저장
            box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update cls_target, cls_mask
            class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class ## 클래스 타겟 텐서를 업데이트.
            class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update iou target and iou mask
            iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :] ##  IOU 타겟 텐서를 업데이트
            iou_mask[b, cell_idx, argmax_anchor_idx, :] = train_params["object_scale"]

    return iou_target.view(bsize, -1, 1), iou_mask.view(bsize, -1, 1), box_target.view(bsize, -1, 4), box_mask.view(bsize, -1, 1), class_target.view(bsize, -1, 1).long(), class_mask.view(bsize, -1, 1)


def yolo_loss(output, target, train_params):
    """
    Build yolo loss

    Arguments:
    output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
    target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data

    delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

    iou_target -- Variable of shape (B, H * W * num_anchors, 1)
    iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
    box_target -- Variable of shape (B, H * W * num_anchors, 4)
    box_mask -- Variable of shape (B, H * W * num_anchors, 1)
    class_target -- Variable of shape (B, H * W * num_anchors, 1)
    class_mask -- Variable of shape (B, H * W * num_anchors, 1)

    Return:
    loss -- yolo overall multi-task loss
    """

    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    iou_target = target[0]
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]

    b, _, num_classes = class_score_batch.size()
    class_score_batch = class_score_batch.view(-1, num_classes)
    class_target = class_target.view(-1)
    class_mask = class_mask.view(-1)

    # ignore the gradient of noobject's target
    class_keep = class_mask.nonzero().squeeze(1)
    class_score_batch_keep = class_score_batch[class_keep, :]
    class_target_keep = class_target[class_keep]

    # if cfg.debug:
    #     print(class_score_batch_keep)
    #     print(class_target_keep)

    # calculate the loss, normalized by batch size.
    box_loss = 1 / b * train_params["coord_scale"] * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * train_params["class_scale"] * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss