import os
import time
import yaml
import torch
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.data import DataLoader

from metrics import DiceLoss
from model.TransResUNet import TResUnet
from data.BKAIDataset import BKAIDataset
from utils import epoch_time, predict, save_config_to_yaml


def eval(model, dataloader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y.long())

            epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(dataloader)

    return epoch_loss


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0
    epoch_acc = 0
    for idx, (x, y) in enumerate(dataloader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y.long())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(dataloader)

    return epoch_loss


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ## Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), config["batch_size"] if config["batch_size"] > 1 else 0, 8])

    ## make save dir
    save_dir = config["save_dir"]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{save_dir}/{current_time}"

    if not os.path.isdir(save_path):
        print(save_path)
        config["save_dir"] = save_path
        os.makedirs(f"{save_path}/weights")
        os.makedirs(f"{save_path}/predict")
    
    save_config_to_yaml(config, save_path)

    ## Load Dataset
    train_dataset = BKAIDataset(config["data_dir"], split="train", size=config["img_size"])
    valid_dataset = BKAIDataset(config["data_dir"], split="valid", size=config["img_size"])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    ## Load pre-trained weight & models
    model = TResUnet(backbone=config["backbone"], num_layers=config["num_layers"])
    model = model.to(device)

    if config["pretrain_weight"] != "":
        model.load_state_dict(torch.load(config["pretrain_weight"]))

    ## Loss Function
    loss_fn = DiceLoss(num_classes=config["num_classes"], crossentropy=config["crossentropy"], give_penalty=config["give_penalty"], penalty_factor=config["penalty_factor"])

    ## Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["initial_lr"], betas=config["betas"])
    if config["scheduler"]:
        div_factor = config["max_lr"] / config["initial_lr"]
        final_div_factor = config["max_lr"] / config["initial_lr"]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        anneal_strategy="cos",
                                                        max_lr=config["max_lr"],
                                                        total_steps=config["epochs"],
                                                        pct_start=config["pct_start"],
                                                        div_factor=div_factor,
                                                        final_div_factor=final_div_factor,
                                                        verbose=True)

    early_stopping_count = 0
    patience = config["early_stopping_patience"]

    ## Train start
    print("\nTrain Start.")
    best_valid_loss = float("inf")
    epochs = config["epochs"]
    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, loss_fn, device)
        valid_loss = eval(model, valid_dataloader, loss_fn, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch [{epoch+1:02}/{epochs}] | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        # data_str += f"\tCurrent Learning Rate: {current_lr} \n"
        data_str += f"\tTrain Loss: {train_loss:.4f} \n"
        data_str += f"\tValid Loss: {valid_loss:.4f} \n"

        if valid_loss < best_valid_loss:
            data_str += f"\tLoss decreased. {best_valid_loss:.4f} ---> {valid_loss:.4f} \n"
            best_valid_loss = valid_loss

            torch.save(model.state_dict(), f"{save_path}/weights/best.pth")
            predict(epoch + 1, config, img_size=config["img_size"], model=model, device=device)

            early_stopping_count = 0

        elif valid_loss > best_valid_loss:
            data_str += f"\tLoss not decreased. {best_valid_loss:.4f} Remaining patience: [{early_stopping_count}/{patience}] \n"
            early_stopping_count += 1

        if config["scheduler"]:
            scheduler.step()

        print(data_str)

        if early_stopping_count == config["early_stopping_patience"]:
            print("Early Stop.\n")
            break