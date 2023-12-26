import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from models.yolov2 import Yolov2
from data.dataset import VOCDataset, detection_collate

from utils.lr_util import adjust_learning_rate
from utils.train_param import read_train_params, save_train_params


def train(model, dataloader, optimizer, epoch, device, train_params):
    model.train()
    loss_temp = 0
    for step, data in enumerate(dataloader):
        image, boxes, gt_classes, num_obj = data

        if device is not None:
            image = image.to(device)
            boxes = boxes.to(device)
            gt_classes = gt_classes.to(device)
            num_obj = num_obj.to(device)

        box_loss, iou_loss, class_loss = model(image, boxes, gt_classes, num_obj, training=True)
        loss = box_loss.mean() + iou_loss.mean() + class_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_temp += loss.item()
        if (step + 1) % train_params["display_interval"] == 0:
            loss_temp /= train_params["display_interval"]
            print(f"[epoch {epoch}][step {step+1}/{len(dataloader)}] loss:{loss_temp:.4f}, lr:{optimizer.param_groups[0]['lr']:.2e}, iou_loss:{iou_loss.mean().item():.4f}, box_loss:{box_loss.mean().item():.4f}, cls_loss:{class_loss.mean().item():.4f}")
            loss_temp = 0

    
def main():
    num_workers = os.cpu_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_params = read_train_params("./config.yaml")

    train_dataset = VOCDataset(root_dir=train_params["data_dir"], image_sets=['train', 'val'], years=["2007", "2012"], img_size=train_params["img_size"])
    valid_dataset = VOCDataset(root_dir=train_params["data_dir"], image_sets=['test'], years=["2007"], img_size=train_params["img_size"])
    train_params["classes"] = train_dataset.classes

    train_dataloader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True, num_workers=num_workers, collate_fn=detection_collate, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_params["batch_size"], num_workers=num_workers, collate_fn=detection_collate, drop_last=True)

    model = Yolov2(train_params).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params["lr"], momentum=train_params["momentum"], weight_decay=train_params["weight_decay"])

    total_epochs = train_params["epochs"]
    scale_range = train_params["scale_range"]
    for epoch in range(1, total_epochs+1):
        print(f"Epoch {epoch}|{total_epochs}")

        if epoch in train_params["decay_lrs"]:
            lr = train_params["decay_lrs"][epoch]
            adjust_learning_rate(optimizer, lr)
            print(f"adjust learning rate to {lr}")

        if train_params["multi_scale"] and epoch in train_params["epoch_scale"]:
            scale_range = train_params["epoch_scale"][epoch]
            print(f"Change scale range to {scale_range}")

        train(model, train_dataloader, optimizer, epoch, device, train_params)
        print()


if __name__ == "__main__":
    main()