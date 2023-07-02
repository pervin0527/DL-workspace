import math
import torch
from torch import nn
from torch.nn import functional as F
from model.bbox_utils import (BoxCoder, box_iou, Matcher, BalancedPositiveNegativeSampler, concat_box_prediction_layers,
                              clip_boxes_to_image, remove_small_boxes, batched_nms, smooth_l1_loss)

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        ## intermediate Layer
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        ## background/foreground score. Objectness Classification
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        ## bbox regression
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        cls_scores = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            cls_scores.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return cls_scores, bbox_reg

class RegionProposalNetwork(nn.Module):
    ## batch_size_per_image: 학습시 sampling 하는 anchor의 수. 논문 기준 256개
    ## positive_fraction: 256개 중 postivie anchor의 비율.
    def __init__(self,
                 anchor_generator,
                 head,
                 positive_iou_thresh,
                 negative_iou_thresh,
                 batch_size_per_image,
                 positive_fraction,
                 pre_nms_top_n,
                 post_nms_top_n,
                 nms_thresh
                 ):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head

        ## bounding box regressor를 학습하기 위해서 encode, decode.
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        ## anchor box와 bounding box간 IoU를 계산
        self.box_similarity = box_iou
        ## 측정한 IoU를 기반으로 ground-truth에 맞는 anchor에 positive, negative label을 적용
        self.proposal_matcher = Matcher(positive_iou_thresh, negative_iou_thresh, allow_low_quality_matches=True)
        ## batch 데이터 당 256개의 anchor를 sampling. 절반은 positive, 나머지는 negative anchor.
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction) ## 256, 0.5

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def forward(self, images, features, targets=None):
        features = list(features.values())
        fg_bg_scores, pred_bbox_deltas = self.head(features)

        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in fg_bg_scores]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        fg_bg_scores, pred_bbox_deltas = concat_box_prediction_layers(fg_bg_scores, pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # remove small bboxes, nms process, get post_nms_top_n target
        boxes, scores = self.filter_proposals(proposals, fg_bg_scores, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)

            # encode parameters based on the bboxes and anchors
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(fg_bg_scores, pred_bbox_deltas, labels, regression_targets)
            losses = {"loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg}

        return boxes, losses

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = box_iou(gt_boxes, anchors_per_image)
                ## calculate index of anchors and gt iou（iou < 0.3은 -1，0.3 < iou < 0.7는 -2）
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                ## target anchor와 ground-truth가 일치하는 것을 선별
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                ## background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                ## -2인 anchor들은 버림.
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)

        return labels, matched_gt_boxes
    
    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        result = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            result.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(result, dim=1)
    
    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        ## 작은 box는 제거하고, nms를 통과해 post_nms_top_n개만 남겨 놓는다.
        num_images = proposals.shape[0]
        device = proposals.device

        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)
        
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            boxes = clip_boxes_to_image(boxes, img_shape)
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)

            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores
    
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # bbox regression loss
        box_loss = smooth_l1_loss(pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds], beta=1 / 9, size_average=False, ) / (sampled_inds.numel())

        # classification loss
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss