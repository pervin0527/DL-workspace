import os
import torch

from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader

from models.yolov1 import Yolov1
from loss_func import YoloLoss

from data.convert_label import classes
from data.augmentation import get_transform
from data.dataset import VOCDetectionDataset

from utils.graph import plot_and_save
from utils.lr_scheduler import LinearWarmupDecayScheduler
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

    save_dir = train_params["save_dir"]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{save_dir}/{current_time}"

    if not os.path.isdir(save_path):
        print(save_path)
        os.makedirs(f"{save_path}/weights")
        os.makedirs(f"{save_path}/plots")
        train_params["save_dir"] = save_path

    save_train_params(save_path, train_params)

    model = Yolov1(grid_size=train_params["grid_size"], num_boxes=train_params["num_boxes"], num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["init_lr"], weight_decay=train_params["weight_decay"])
    loss_func = YoloLoss()

    if train_params["use_scheduler"]:
        scheduler = LinearWarmupDecayScheduler(optimizer, train_params["init_lr"], train_params["max_lr"], train_params["min_lr"], train_params["epochs"], train_params["warmup_epochs"])

    train_losses = []
    valid_losses = []
    learning_rates = []
    min_valid_loss = float('inf')
    total_epochs = train_params["epochs"]
    for epoch in range(total_epochs):
        if train_params["use_scheduler"]:
            current_lr = scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)
            print(f"\nEpoch : [{epoch + 1} | {total_epochs}] \tCurrent Learning Rate: {current_lr:.6f}")
        else:
            print(f"\nEpoch : [{epoch + 1} | {total_epochs}]")

        train_loss = train(train_dataloader, model, optimizer, loss_func, device)
        print(f"\tTrain Loss : {train_loss:.4f}")

        pred_boxes, target_boxes, valid_loss = validation(valid_dataloader, model, loss_func, device, iou_thres=0.5, thres=0.4)
        print(f"\tValid Loss : {valid_loss:.4f}")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < min_valid_loss:
            print(f"Model saved at epoch {epoch+1} with validation loss {min_valid_loss:.4f} --> {valid_loss:.4f}")
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{save_path}/weights/ep_{epoch+1}_{valid_loss:.4f}.pth')
        else:
            print(f"Validation loss {valid_loss:.4f} did not decrease. {min_valid_loss:.4f}")

        if train_params["use_scheduler"]:
            scheduler.step()

        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"\tTrain mAP: {mean_avg_prec:.4f}")

    plot_and_save(train_losses, 'Training Loss', 'Loss', f'{save_path}/plots/train_loss.png')
    plot_and_save(valid_losses, 'Validation Loss', 'Loss', f'{save_path}/plots/valid_loss.png')
    plot_and_save(learning_rates, 'Learning Rate', 'Learning Rate', f'{save_path}/plots/learning_rate.png')

if __name__ == "__main__":
    main()