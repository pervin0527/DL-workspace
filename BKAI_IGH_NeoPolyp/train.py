import os
import time
import yaml
import torch
import shutil

from torch.utils.data import DataLoader

from model import TResUnet
from metrics import SegmentationLosses
from data.BKAI_dataset import BKAIDatasetV1, BKAIDatasetV2
from utils import epoch_time, predict, calculate_weigths_labels


def eval(model, dataloader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(dataloader)

    return epoch_loss


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for idx, (x, y) in enumerate(dataloader):
        # x, y = data["image"], data["mask"]
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(dataloader)

    return epoch_loss


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if os.path.exists("./predict"):
        shutil.rmtree("./predict")

    ## Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), config["batch_size"] if config["batch_size"] > 1 else 0, 8])

    ## Load Dataset
    train_dataset = BKAIDatasetV1(config["data_dir"], threshold=config["mask_threshold"], split="train") ## mean=config["means"], std=config["stds"]
    valid_dataset = BKAIDatasetV1(config["data_dir"], threshold=config["mask_threshold"], split="valid") ## mean=config["means"], std=config["stds"]

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config["batch_size"],
                                  shuffle=True,
                                  num_workers=num_workers)
    
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    ## Calculate class weights
    # weight = calculate_weigths_labels(config["data_dir"], train_dataloader, 3)
    # weight = torch.from_numpy(weight.astype(np.float32))

    ## Load pre-trained weights & models
    model = TResUnet()
    model = model.to(device)

    ## Loss func & Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]), betas=(0.9, 0.999))
    loss_fn = SegmentationLosses(weight=None, cuda=device).build_loss(mode="ce")
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config["decay_term"], T_mult=1, eta_min=config["min_lr"])  # 10 에폭마다 Warm Restart, 최소 LR는 0.00001

    ## Epochs
    print("\nStart training")

    best_valid_loss = float("inf")
    early_stopping_count = 0
    for epoch in range(config["epochs"]):
        start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, loss_fn, device)
        valid_loss = eval(model, valid_dataloader, loss_fn, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if valid_loss < best_valid_loss:
            if not os.path.isdir(config["save_dir"]):
                os.makedirs(config["save_dir"])

            early_stopping_count = 0
            best_valid_loss = valid_loss

            path = config["save_dir"] + "/" + config["save_name"]
            torch.save(model.state_dict(), path)

        elif valid_loss > best_valid_loss:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tCurrent Learning Rate: {current_lr} \n"  # learning rate 출력
        data_str += f"\tTrain Loss: {train_loss:.4f} \n"
        data_str += f"\tValid Loss: {valid_loss:.4f} \n"
        print(data_str)

        predict(epoch=epoch, data_dir=config["data_dir"], img_size=config["img_size"], model=model, device=device)
        # scheduler.step()

        if early_stopping_count == config["early_stopping_patience"]:
            data_str = "Early stopping.\n"
            break
