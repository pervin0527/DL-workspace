import torch
import numpy as np
from torchvision.ops import nms
from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox

class ProposalCreator:
    """
    해당 객체의 '__call__' 메서드는 anchor들에 predict bbox offset을 적용해 object detection proposal을 생성한다.
    매개변수들을 사용해 nms함수로 전달하고, nms 처리 단계에서 유지하려는 bbox의 수를 제어한다.
    음수값이 전달되면 모든 bbox를 사용하거나 nms에서 반환한 모든 bbox를 유지한다.
    """
    def __init__(self, parent_model, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16):
        """
        args:
            nms_thresh : nms함수에 적용될 threshold.
            n_train_pre_nms : 학습 단계에서 nms 통과 전에 남겨둘 최고 점수의 bbox의 수.
            n_train_post_nms : 학습 단계에서 nms를 통과한 후 유지할 최고 점수의 bbox의 수.
            n_test_pre_nms : 테스트 단계에서 nms 통과 전에 남겨둘 최고 점수의 bbox의 수.
            n_test_post_nms : 테스트 단계에서 nms 통과 후 유지하려는 최고 점수의 bbox의 수.

        """
        self.parent_model = parent_model
        self.nms_thres = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        """
        입력값은 ndarray propose region of interests.
        loc, score, anchor는 동일한 index로 indexing될 때 동일한 anchor를 참조하게 된다.
        R은 전체 anchor의 수로, image의 height, width의 픽셀당 적용되는 anchor의 수를 곱한 것과 같다.

        args:
            loc(array) : 예측된 anchor에 대한 scaling과 offset
            score(array) : 예측된 anchor에 대한 foreground probability
            anchor(array) : anchor의 좌표.
        """

        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        ## bbox 변환을 통해 anchor를 proposal로 변환한다.
        roi = loc2bbox(anchor, loc)
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        ## threshold 보다 낮은 bbox는 제거.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        ## 최고점 -> 최저점 순으로 정렬.
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        keep = nms(torch.from_numpy(roi).cuda(), torch.from_numpy(score).cuda(), self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]

        roi = roi[keep.cpu().numpy()]
        
        return roi