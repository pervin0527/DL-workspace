import time
import torch
import numpy as np
import torch.utils.tensorboard
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from model import YoloV3
from dataset import DetectionDataset
from utils import init_weights_normal, xywh2xyxy, ap_per_class, non_max_suppression, get_batch_statistics

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)

    epochs = 1000
    batch_size = 64
    image_size = 416
    
    learning_rate = 0.001
    step_size = 10
    gamma = 0.8
    gradient_accumulation = 1

    iou_thres = 0.5
    nms_thres = 0.5
    conf_thres = 0.5

    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    save_dir = "/home/pervinco/Models/YoloV3"

    writer = torch.utils.tensorboard.SummaryWriter(f"{save_dir}/logs")

    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DetectionDataset(data_dir=data_dir, set_name="train", annot_type="voc", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    classes = train_dataset.classes

    valid_transform = transforms.Compose([transforms.ToTensor()])
    valid_dataset = DetectionDataset(data_dir=data_dir, set_name="valid", annot_type="voc", transform=valid_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    model = YoloV3(image_size=image_size, num_classes=len(classes)).to(device)
    model.apply(init_weights_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_log = tqdm(total=0, position=2, bar_format="{desc}", leave=False)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()

        for batch_idx, (_, images, targets) in enumerate(tqdm(train_dataloader, desc="Batch", leave=False)):
            step = len(train_dataloader) * epoch + batch_idx

            images = images.to(device)
            targets = targets.to(device)

            loss, outputs = model(images, targets)
            loss.backward()

            if step % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_log.set_description_str('Loss: {:.6f}'.format(loss.item()))

        # Tensorboard에 훈련 과정 기록
        tensorboard_log = []
        for i, yolo_layer in enumerate(model.yolo_layers):
            writer.add_scalar('loss_bbox_{}'.format(i + 1), yolo_layer.metrics['loss_bbox'], step)
            writer.add_scalar('loss_conf_{}'.format(i + 1), yolo_layer.metrics['loss_conf'], step)
            writer.add_scalar('loss_cls_{}'.format(i + 1), yolo_layer.metrics['loss_cls'], step)
            writer.add_scalar('loss_layer_{}'.format(i + 1), yolo_layer.metrics['loss_layer'], step)
        writer.add_scalar('total_loss', loss.item(), step)

    # lr scheduler의 step을 진행
    scheduler.step()

    labels = []
    sample_metrics = []  # List[Tuple] -> [(TP, confs, pred)]
    entire_time = 0
    for _, images, targets in tqdm.tqdm(valid_dataloader, desc='Evaluate method', leave=False):
        if targets is None:
            continue

        # Extract labels
        labels.extend(targets[:, 1].tolist())

        # Rescale targets
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= image_size

        # Predict objects
        start_time = time.time()
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            outputs = non_max_suppression(outputs, conf_thres, nms_thres)
        entire_time += time.time() - start_time

        # Compute true positives, predicted scores and predicted labels per batch
        sample_metrics.extend(get_batch_statistics(outputs, targets, iou_thres))

    # Concatenate sample statistics
    if len(sample_metrics) == 0:
        true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
    else:
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    # Compute AP
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Compute inference time and fps
    inference_time = entire_time / len(valid_dataset)
    fps = 1 / inference_time

    # Export inference time to miliseconds
    inference_time *= 1000

    # Tensorboard에 평가 결과 기록
    writer.add_scalar('val_precision', precision.mean(), epoch)
    writer.add_scalar('val_recall', recall.mean(), epoch)
    writer.add_scalar('val_mAP', AP.mean(), epoch)
    writer.add_scalar('val_f1', f1.mean(), epoch)

    torch.save(model.state_dict(), f"{save_dir}/yolov3.pth")