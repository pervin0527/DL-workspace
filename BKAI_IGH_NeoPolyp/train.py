import os
import cv2
import time
import yaml
import torch
import numpy as np
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt

from random import randint
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import TResUnet
from data.BKAI_dataset import BKAI_Dataset
from metrics import DiceLoss, multi_class_dice_coefficient
from utils import epoch_time


def predict(model):
    dir = config["data_dir"]
    size = config["img_size"]

    with open(f"{dir}/valid.txt", "r") as f:
        files = f.readlines()
    
    idx = randint(0, len(files)-1)
    file = files[idx]
    file = file.strip()

    image = cv2.imread(f"{dir}/train/{file}.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(f"{dir}/train_gt/{file}.jpeg")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (size, size))
    x = np.transpose(x, (2, 0, 1)) ## H, W, C -> C, H, W
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x) / 255.0
    x = x.to(device)

    y_pred = model(x)
    pred_mask = torch.argmax(y_pred[0], dim=0).cpu().numpy()

    # y_pred = torch.cat(y_pred, dim=1)
    # _, pred_mask = torch.max(y_pred, dim=1)
    # pred_mask = pred_mask.squeeze(0).cpu().numpy()

    color_decoded = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    color_decoded[pred_mask == 0] = [0, 0, 0]
    color_decoded[pred_mask == 1] = [255, 0, 0]
    color_decoded[pred_mask == 2] = [0, 255, 0]
    color_decoded = cv2.resize(color_decoded, (mask.shape[1], mask.shape[0]))

    overlayed = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(overlayed)
    plt.title('Original Mask')

    plt.subplot(1, 2, 2)
    plt.imshow(color_decoded)
    plt.title('Decoded Label')

    plt.savefig("./predict.png")
    plt.close()


def eval(model, dataloader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += multi_class_dice_coefficient(y_pred, y)

    epoch_loss = epoch_loss / len(dataloader)
    epoch_acc = epoch_acc / len(dataloader)
    predict(model)

    return epoch_loss, epoch_acc


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for idx, (x, y) in enumerate(dataloader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += multi_class_dice_coefficient(y_pred, y)

    epoch_loss = epoch_loss / len(dataloader)
    epoch_acc = epoch_acc / len(dataloader)

    return epoch_loss, epoch_acc


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ## Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), config["batch_size"] if config["batch_size"] > 1 else 0, 8])

    # train_transform = A.Compose([A.Rotate(limit=35, p=0.3),
    #                              A.HorizontalFlip(p=0.3),
    #                              A.VerticalFlip(p=0.3),
    #                              A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)])

    ## Load Dataset
    train_dataset = BKAI_Dataset(config["data_dir"], "train", config["img_size"], config["mask_threshold"])
    valid_dataset = BKAI_Dataset(config["data_dir"], "valid", config["img_size"], config["mask_threshold"])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    ## Calculate class weights
    class_counts = config["class_distribution"]
    total_counts = sum(class_counts)
    weights = [total_counts/class_count for class_count in class_counts]
    weights = [w/sum(weights) for w in weights]
    print(weights)

    ## Load pre-trained weights & models
    model = TResUnet()
    model = model.to(device)

    ## Loss func & Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]), betas=(0.9, 0.999))
    loss_fn = DiceLoss()
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config["decay_term"], T_mult=1, eta_min=config["min_lr"])  # 10 에폭마다 Warm Restart, 최소 LR는 0.00001

    ## Epochs
    best_valid_loss = float('inf')
    for epoch in range(config["epochs"]):
        start_time = time.time()
        train_loss, train_acc = train(model, train_dataloader, optimizer, loss_fn, device)
        valid_loss, valid_acc = eval(model, valid_dataloader, loss_fn, device)
        current_lr = optimizer.param_groups[0]['lr']

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tCurrent Learning Rate: {current_lr} \n"  # learning rate 출력
        data_str += f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} \n"
        data_str += f"\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} \n"
        print(data_str)

        if valid_loss < best_valid_loss:
            if not os.path.isdir(config["save_dir"]):
                os.makedirs(config["save_dir"])

            best_valid_loss = valid_loss

            path = config["save_dir"] + '/' + config["save_name"]
            torch.save(model.state_dict(), path)

        # scheduler.step()

