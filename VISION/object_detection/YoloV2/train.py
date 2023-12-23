import os
import torch

from tqdm import tqdm
from datetime import datetime
from torch import optim
from torch.utils.data import DataLoader

from loss import YoloLoss
from models.yolov2 import YoloV2
from data.dataset import VOCDataset, custom_collate_fn

from utils.train_param import read_train_params

def eval(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_loss_coord = 0.0
    total_loss_conf = 0.0
    total_loss_cls = 0.0
    with torch.no_grad():  # 기울기 계산을 중지
        for image, label in tqdm(dataloader, desc="Evaluate", leave=False):
            image = image.to(device)
            pred = model(image)

            loss, loss_coord, loss_conf, loss_cls = criterion(pred, label)

            total_loss += loss.item() * image.size(0)
            total_loss_coord += loss_coord.item() * image.size(0)
            total_loss_conf += loss_conf.item() * image.size(0)
            total_loss_cls += loss_cls.item() * image.size(0)

    average_loss = total_loss / len(dataloader.dataset)
    average_loss_coord = total_loss_coord / len(dataloader.dataset)
    average_loss_conf = total_loss_conf / len(dataloader.dataset)
    average_loss_cls = total_loss_cls / len(dataloader.dataset)

    return average_loss, average_loss_coord, average_loss_conf, average_loss_cls


def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_loss_coord = 0.0
    total_loss_conf = 0.0
    total_loss_cls = 0.0
    for image, label in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()
        pred = model(image.to(device))

        loss, loss_coord, loss_conf, loss_cls = criterion(pred, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image.size(0)
        total_loss_coord += loss_coord.item() * image.size(0)
        total_loss_conf += loss_conf.item() * image.size(0)
        total_loss_cls += loss_cls.item() * image.size(0)

    average_loss = total_loss / len(dataloader.dataset)
    average_loss_coord = total_loss_coord / len(dataloader.dataset)
    average_loss_conf = total_loss_conf / len(dataloader.dataset)
    average_loss_cls = total_loss_cls / len(dataloader.dataset)

    return average_loss, average_loss_coord, average_loss_conf, average_loss_cls


def main():
    num_workers = os.cpu_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_params = read_train_params("./config.yaml")

    train_dataset = VOCDataset(root_path=train_params["data_dir"], year="2012", mode="train", image_size=train_params["img_size"])
    valid_dataset = VOCDataset(root_path=train_params["data_dir"], year="2012", mode="val", image_size=train_params["img_size"])

    train_dataloader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, drop_last=True, collate_fn=custom_collate_fn, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_params["batch_size"], drop_last=True, collate_fn=custom_collate_fn, num_workers=num_workers)

    model = YoloV2(num_classes=20).cuda()
    criterion = YoloLoss(num_classes=20, anchors=model.anchors, reduction=32)
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params["init_lr"], momentum=train_params["momentum"], weight_decay=train_params["weight_decay"])

    epochs = train_params["epochs"]
    lr_schedule = train_params["lr_schedule"]

    for epoch in range(epochs):
        if str(epoch) in lr_schedule.keys():
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[str(epoch)]
        
        print(f"\nEpoch : [{epoch + 1} | {epochs}]")
        
        train_loss, train_bbox_loss, train_conf_loss, train_cls_loss = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Train Loss : {train_loss:.4f}, Bbox Loss : {train_bbox_loss}, Conf Loss : {train_conf_loss}, Classification Loss : {train_cls_loss}")

        valid_loss, valid_bbox_loss, valid_conf_loss, valid_cls_loss = eval(model, valid_dataloader, criterion, device)
        print(f"valid Loss : {valid_loss:.4f}, Bbox Loss : {valid_bbox_loss}, Conf Loss : {valid_conf_loss}, Classification Loss : {valid_cls_loss}")


if __name__ == "__main__":
    main()