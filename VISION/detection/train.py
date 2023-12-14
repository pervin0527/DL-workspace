import yaml
import torch

from tqdm import tqdm

from torch.utils.data import DataLoader

from models.yolov1 import Yolov1
from loss_func import YoloLoss

from data.convert_label import classes
from data.augmentation import get_transform
from data.dataset import VOCDetectionDataset

from utils.train_param import read_train_params, save_train_params
from utils.detection_utils import cellboxes_to_boxes, non_max_suppression, mean_average_precision


def validation(dataloader, model, loss_func, device, iou_thres=0.45, thres=0.45, box_format="midpoint"):
    model.eval()

    i = 0
    valid_loss = 0.0
    all_pred_boxes, all_true_boxes = [], []
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            predictions = model(x) ## [grid_size, grid_size, (num_boxes * 5 + num_classes)]
            loss = loss_func(predictions, y)

            valid_loss += loss.item() * x.size(0)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(y)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=iou_thres, threshold=thres, box_format=box_format)

            for nms_box in nms_boxes:
                all_pred_boxes.append([i] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > thres:
                    all_true_boxes.append([i] + box)

            i += 1

    valid_loss /= len(dataloader.dataset)

    return all_pred_boxes, all_true_boxes, valid_loss


def train(dataloader, model, optimizer, loss_func, device):
    model.train()
    
    train_loss = 0.0
    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)

        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    train_loss /= len(dataloader.dataset)

    return train_loss


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_params = read_train_params("./config.yaml")

    train_transform = get_transform(True, 448)
    valid_transform = get_transform(False, 448)

    train_dataset = VOCDetectionDataset("./data/train.txt", grid_scale=train_params["grid_size"], num_boxes=train_params["num_boxes"], num_classes=len(classes), transform=train_transform)
    valid_dataset = VOCDetectionDataset("./data/test.txt", grid_scale=train_params["grid_size"], num_boxes=train_params["num_boxes"], num_classes=len(classes), transform=valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_params["batch_size"], shuffle=False)

    model = Yolov1(grid_size=train_params["grid_size"], num_boxes=train_params["num_boxes"], num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["learning_rate"], weight_decay=train_params["weight_decay"])
    loss_func = YoloLoss()

    epochs = train_params["epochs"]
    for epoch in range(epochs):
        train_loss = train(train_dataloader, model, optimizer, loss_func, device)
        print(train_loss)

        pred_boxes, target_boxes, valid_loss = validation(valid_dataloader, model, loss_func, device, iou_thres=0.5, thres=0.4)
        print(pred_boxes, target_boxes, valid_loss)

        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"Train mAP: {mean_avg_prec}")

if __name__ == "__main__":
    main()