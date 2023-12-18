import os
import torch

from tqdm import tqdm
from datetime import datetime

from torch import optim
from torchsummary import summary
from torch.utils.data import DataLoader

from data.dataset import VOCDataset
from data.augmentation import get_transform

from loss import YoloLoss

from models.yolov1 import Yolov1

from utils.graph import plot_and_save
from utils.lr_scheduler import LinearWarmupDecayScheduler
from utils.train_param import read_train_params, save_train_params
from utils.detection_utils import mean_average_precision, get_bboxes


def eval(dataloader, model, criterion, device):
    model.eval()

    eval_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            eval_loss += loss.item() * x.size(0)
    
    eval_loss /= len(dataloader.dataset)

    return eval_loss


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
    num_workers = os.cpu_count()
    train_params = read_train_params("./config.yaml")

    train_transform = get_transform(is_train=True, img_size=train_params["img_size"])
    valid_transform = get_transform(is_train=False, img_size=train_params["img_size"])

    train_dataset = VOCDataset("data/train.csv", transform=train_transform)
    test_dataset = VOCDataset("data/test.csv", transform=valid_transform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_params["batch_size"], shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(dataset=test_dataset, batch_size=train_params["batch_size"], num_workers=num_workers)

    model = Yolov1(grid_size=7, num_boxes=2, num_classes=20)
    optimizer = optim.Adam(model.parameters(), lr=train_params["init_lr"], weight_decay=train_params["weight_decay"])
    loss_func = YoloLoss()
    summary(model, input_size=(3, train_params["img_size"], train_params["img_size"]), device="cpu")
    model.to(device)

    if train_params["use_scheduler"]:
        scheduler = LinearWarmupDecayScheduler(optimizer, train_params["init_lr"], train_params["max_lr"], train_params["min_lr"], train_params["epochs"], train_params["warmup_epochs"])

    save_dir = train_params["save_dir"]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{save_dir}/{current_time}"

    if not os.path.isdir(save_path):
        print(save_path)
        os.makedirs(f"{save_path}/weights")
        os.makedirs(f"{save_path}/plots")
        train_params["save_dir"] = save_path
    
    save_train_params(save_path, train_params)

    train_losses = []
    valid_losses = []
    learning_rates = []
    max_mAP = 0
    total_epochs = train_params["epochs"]
    for epoch in range(total_epochs):
        print(f"\nEpoch : [{epoch + 1} | {total_epochs}]")

        train_loss = train(train_dataloader, model, optimizer, loss_func, device)
        pred_boxes, target_boxes = get_bboxes(train_dataloader, model, threshold=train_params["threshold"], iou_threshold=train_params["iou_threshold"])
        train_mAP = mean_average_precision(pred_boxes, target_boxes, iou_threshold=train_params["iou_threshold"], box_format="midpoint")
        print(f"Train Loss : {train_loss:.4f}, Train mAP : {train_mAP:.4f}")

        valid_loss = eval(valid_dataloader, model, loss_func, device)
        pred_boxes, target_boxes = get_bboxes(valid_dataloader, model, threshold=train_params["threshold"], iou_threshold=train_params["iou_threshold"])
        valid_mAP = mean_average_precision(pred_boxes, target_boxes, iou_threshold=train_params["iou_threshold"], box_format="midpoint")
        print(f"Valid Loss : {valid_loss:.4f}, Valid mAP : {valid_mAP:.4f}")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_mAP > max_mAP:
            print(f"Model saved at epoch {epoch+1} with mAP {max_mAP:.4f} --> {valid_mAP:.4f}")
            max_mAP = valid_mAP
            torch.save(model.state_dict(), f'{save_path}/weights/ep_{epoch+1}_{valid_mAP:.4f}.pth')
        else:
            print(f"mAP {valid_mAP:.4f} did not increase. {max_mAP:.4f}")

        if train_params["use_scheduler"]:
            scheduler.step()

    plot_and_save(train_losses, 'Training Loss', 'Loss', f'{save_path}/plots/train_loss.png')
    plot_and_save(valid_losses, 'Validation Loss', 'Loss', f'{save_path}/plots/valid_loss.png')
    plot_and_save(learning_rates, 'Learning Rate', 'Learning Rate', f'{save_path}/plots/learning_rate.png')


if __name__ == "__main__":
    main()