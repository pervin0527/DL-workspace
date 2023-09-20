import os
import time
import yaml
import torch

from datetime import datetime
from torch.utils.data import DataLoader

from model import TResUnet
from metrics import DiceLoss, DiceCELoss
from data.BKAI_dataset import BKAIDataset
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

    ## Load Dataset
    train_dataset = BKAIDataset(config["data_dir"], split="train", size=config["img_size"], threshold=config["mask_threshold"])
    valid_dataset = BKAIDataset(config["data_dir"], split="valid", size=config["img_size"], threshold=config["mask_threshold"])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    ## Load pre-trained weights & models
    model = TResUnet(backbone=config["backbone"])
    model = model.to(device)

    ## Optimizer
    if config["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["basic_lr"]), betas=config["betas"])
    elif config["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config["basic_lr"]), momentum=config["momentum"])

    ## Loss Function
    class_weights = torch.tensor([0.5, 1.0, 5.0]).to(device)

    if config["loss_fn"].lower() == "ce":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif config["loss_fn"].lower() == "dice":
        loss_fn = DiceLoss()
    elif config["loss_fn"].lower() == "cedice":
        loss_fn = DiceCELoss()


    ## LR Scheduler
    if config["cosine_annealing"]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config["start_lr"]

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=config["min_lr"])

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

    ## Train start
    print("\nTrain Start.")
    best_valid_loss = float("inf")
    early_stopping_count = 0
    for epoch in range(config["epochs"]):
        start_time = time.time()
        train_loss = train(model, train_dataloader, optimizer, loss_fn, device)
        valid_loss = eval(model, valid_dataloader, loss_fn, device)
        current_lr = optimizer.param_groups[0]["lr"]

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tCurrent Learning Rate: {current_lr} \n"  # learning rate 출력
        data_str += f"\tTrain Loss: {train_loss:.4f} \n"
        data_str += f"\tValid Loss: {valid_loss:.4f} \n"

        if valid_loss < best_valid_loss:
            early_stopping_count = 0
            data_str += f"\tLoss decreased. {best_valid_loss:.4f} ---> {valid_loss:.4f} \n"
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{save_path}/weights/best.pth")

        elif valid_loss > best_valid_loss:
            patience = config["early_stopping_patience"]
            data_str += f"\tLoss not decreased. {best_valid_loss:.4f} Remaining: [{early_stopping_count}/{patience}] \n"
            early_stopping_count += 1

        print(data_str)
        predict(epoch, config, img_size=config["img_size"], model=model, device=device)

        if config["cosine_annealing"]:
            scheduler.step()


        if early_stopping_count == config["early_stopping_patience"]:
            print("Early Stop.\n")
            break
