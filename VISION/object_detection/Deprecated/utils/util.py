import os
import yaml
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler

from models.util import bbox_iou


class LinearWarmupDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, init_lr, max_lr, min_lr, total_epochs, warmup_epochs, last_epoch=-1):
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
        super(LinearWarmupDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.total_epochs <= self.warmup_epochs:
            raise ValueError("Total epochs must be greater than warmup epochs")

        if self.last_epoch < self.warmup_epochs:
            # 워밍업 단계 - init_lr에서 max_lr까지 선형 증가
            lr = self.init_lr + (self.max_lr - self.init_lr) * (self.last_epoch / self.warmup_epochs)
        else:
            # 선형 감소 단계 - max_lr에서 min_lr까지 선형 감소
            decay_epochs = self.total_epochs - self.warmup_epochs
            if decay_epochs > 0:
                lr = self.max_lr - (self.max_lr - self.min_lr) * ((self.last_epoch - self.warmup_epochs) / decay_epochs)
            else:
                lr = self.min_lr
        return [lr for base_lr in self.base_lrs]

def read_yaml(file_path):
    with open(file_path, "r") as f:
        contents = yaml.safe_load(f)
    
    return contents


def save_yaml(save_dir, contents):
    with open(f"{save_dir}/config.yaml", "w") as f:
        yaml.dump(contents, f)


def make_log_dir(save_dir, record_contents=None):
    if not os.path.isdir(save_dir):
        os.makedirs(f"{save_dir}/weights")
        os.makedirs(f"{save_dir}/logs")
        
        if record_contents is not None:
            save_yaml(save_dir, record_contents)


def get_batch_statistics(outputs, targets, iou_threshold):
    """Compute true positives, predicted scores and predicted labels per batch."""
    batch_metrics = []
    for i, output in enumerate(outputs):

        if output is None:
            continue

        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    
    return batch_metrics


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """
    Compute the average precision, given the Precision-Recall curve.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    # for c in tqdm(unique_classes, desc="Compute AP", leave=False):
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")