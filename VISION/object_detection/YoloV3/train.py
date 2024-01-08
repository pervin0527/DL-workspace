import os
import time
import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from models.yolov3 import YoloV3
from data.dataset import YoloDataset

from utils.util import read_yaml, save_yaml, make_log_dir, LinearWarmupDecayScheduler


def eval(model, dataloader):
    model.eval()
    valid_loss, valid_bbox_loss, valid_conf_loss, valid_cls_loss = 0, 0, 0, 0
    for paths, images, targets in tqdm(dataloader, desc="Valid", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            loss, outputs = model(images, targets)
        
        for idx, head in enumerate(model.heads):
            valid_bbox_loss += head.metrics["loss_bbox"]
            valid_conf_loss += head.metrics["loss_conf"]
            valid_cls_loss += head.metrics["loss_cls"]
            valid_loss += head.metrics["loss_layer"]
    
    valid_bbox_loss /= len(dataloader.dataset)
    valid_conf_loss /= len(dataloader.dataset)
    valid_cls_loss /= len(dataloader.dataset)
    valid_loss /= len(dataloader.dataset)

    return valid_bbox_loss, valid_conf_loss, valid_cls_loss, valid_loss


def train(model, dataloader, optimizer):
    model.train()
    train_loss, train_bbox_loss, train_conf_loss, train_cls_loss = 0, 0, 0, 0
    for paths, images, targets in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss, outputs = model(images, targets)
        loss.backward()
        optimizer.step()

        for idx, head in enumerate(model.heads):
            train_bbox_loss += head.metrics["loss_bbox"]
            train_conf_loss += head.metrics["loss_conf"]
            train_cls_loss += head.metrics["loss_cls"]
            train_loss += head.metrics["loss_layer"]
    
    train_bbox_loss /= len(dataloader.dataset)
    train_conf_loss /= len(dataloader.dataset)
    train_cls_loss /= len(dataloader.dataset)
    train_loss /= len(dataloader.dataset)

    return train_bbox_loss, train_conf_loss, train_cls_loss, train_loss


def main():
    now = datetime.now().strftime('%y%m%d_%H%M%S')
    print(now)

    train_path = cfg['train']
    valid_path = cfg['valid']
    class_names = cfg['names']
    print(f"{train_path}\n{valid_path}\n{class_names}")

    train_dataset = YoloDataset(train_path, cfg["img_size"], augment=True, multiscale=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=num_workers, collate_fn=train_dataset.collate_fn)

    valid_dataset = YoloDataset(valid_path, cfg["img_size"])
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg["batch_size"], num_workers=num_workers, collate_fn=valid_dataset.collate_fn)


    model = YoloV3(anchors=cfg["anchors"], img_size=cfg["img_size"], num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = LinearWarmupDecayScheduler(optimizer, init_lr=cfg["init_lr"], min_lr=cfg["min_lr"], max_lr=cfg["max_lr"], total_epochs=cfg["epochs"], warmup_epochs=cfg["warmup_epochs"])

    epochs = cfg["epochs"]
    min_valid_loss = float('inf')

    save_path = cfg["save_dir"] + f"/{now}"
    make_log_dir(save_dir=save_path, record_contents=True)
    for epoch in range(epochs):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch [{epoch} | {epochs}], Current Learning Rate: {current_lr:.6f}")

        train_bbox_loss, train_conf_loss, train_cls_loss, train_loss = train(model, train_dataloader, optimizer)
        print(f"Train | bbox_loss : {train_bbox_loss:.4f}, conf_loss : {train_conf_loss:.4f}, cls_loss : {train_cls_loss:.4f}, total_loss : {train_loss:.4f}")
        
        valid_bbox_loss, valid_conf_loss, valid_cls_loss, valid_loss = eval(model, valid_dataloader)
        print(f"Valid | bbox_loss : {valid_bbox_loss:.4f}, conf_loss : {valid_conf_loss:.4f}, cls_loss : {valid_cls_loss:.4f}, total_loss : {valid_loss:.4f}")

        if valid_loss < min_valid_loss:
            print(f"Model saved at epoch {epoch+1} with validation loss {min_valid_loss:.4f} --> {valid_loss:.4f}")
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{save_path}/weights/best.pth')
        else:
            print(f"Validation loss {valid_loss:.4f} did not decrease. {min_valid_loss:.4f}")

        scheduler.step()


if __name__ == "__main__":
    num_workers = os.cpu_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = read_yaml("./config.yaml")

    main()