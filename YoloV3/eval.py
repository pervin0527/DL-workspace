import time
import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data
from config import test_cfg
from utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class


def evaluate(model, dataloader, device):
    model.eval()
    labels = []
    sample_metrics = []  # List[Tuple] -> [(TP, confs, pred)]
    for file_name, images, targets in tqdm(dataloader, desc='Evaluate method', leave=False):
        if targets is None:
            continue

        labels.extend(targets[:, 1].tolist())

        ## Rescale BBox
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= test_cfg.img_size

        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = non_max_suppression(outputs, test_cfg.conf_thres, test_cfg.nms_thres)
        
        sample_metrics.extend(get_batch_statistics(outputs, targets, test_cfg.iou_thres))
        
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class