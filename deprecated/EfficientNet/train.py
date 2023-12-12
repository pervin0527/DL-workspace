import os
import yaml
import torch

from torch import nn
from torch.utils.data import DataLoader

from data.dataloader import MyDataset
from models.model import EfficientNet
from utils.utils import make_save_dir
from torch.utils.tensorboard import SummaryWriter

def valid(model, dataloader, loss_func):
    model.eval()
    valid_loss, valid_acc = 0, 0
    
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            Y_PRED = model(X)
            loss = loss_func(Y_PRED, Y)

            valid_loss += loss.item() * X.size(0)

            _, Y_PRED = torch.max(Y_PRED, 1)
            valid_acc += (Y_PRED == Y).sum().item()

    valid_acc /= len(dataloader.dataset)
    valid_loss /= len(dataloader.dataset)

    return valid_loss, valid_acc


def train(model, dataloader, criterion, optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for i, (X, Y) in enumerate(dataloader):
        X = X.to(device)
        Y = Y.to(device)

        Y_PRED = model(X)
        loss = criterion(Y_PRED, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

        _, Y_PRED = torch.max(Y_PRED, 1)
        train_acc += (Y_PRED == Y).sum().item()

    train_loss /= len(dataloader.dataset)
    train_acc = train_acc / len(dataloader.dataset)

    return train_loss, train_acc
        

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
           config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    save_path = config["save_dir"]
    make_save_dir(save_path)
    writer = SummaryWriter(log_dir=f"{save_path}/logs")


    model_name= config["model_name"]
    if config["pretrained"]:
        model = EfficientNet.from_pretrained(model_name, advprop=config["advprop"])
        print(f"=> using pre-trained model {model_name}")
    else:
        model = EfficientNet.from_name(model_name)
        print(f"=> creating model {model_name}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), config["learning_rate"],
                                momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    
    img_size = EfficientNet.get_image_size(model_name)
    train_dataset = MyDataset(data_dir=config["data_dir"], set_name=config["train_set"], img_size=img_size)
    valid_dataset = MyDataset(data_dir=config["data_dir"], set_name=config["valid_set"], img_size=img_size)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"])
    
    start_epoch = config["start_epoch"]
    retraining = config["resume"]
    if retraining:
         if os.path.isfile(retraining):
              print(f"=> Loading checkpoint {retraining}")
              checkpoint = torch.load(retraining)

              start_epoch = checkpoint["epoch"]
              best_acc1 = checkpoint["best_acc1"]
              best_acc1 = best_acc1.to(device)

              model.load_state_dict(checkpoint["state_dict"])
              optimizer.load_state_dict(checkpoint["optimizer"])

              print(f"=> loaded checkpoint {retraining} epoch : {start_epoch}")
    else:
         print(f"=> no checkpoint found at {retraining}")

    EARLY_STOP_PATIENCE = config["early_stop_patience"]
    best_acc, best_loss = 0, 0
    epochs = config["epochs"]
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
        valid_loss, valid_acc = valid(model, valid_dataloader, criterion)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", valid_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", valid_acc, epoch)

        print(f"\nEPOCH[{epoch+1} | {epochs}]")
        print(f"Train Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")
        print(f"Valid Loss : {valid_loss:.4f}, Valid Acc : {valid_acc:.4f}")

        writer.add_scalars("Loss", {"train_loss" : train_loss, "valid_loss" : valid_loss}, epoch)
        writer.add_scalars("Accuracy", {"train_accuracy" : train_acc, "valid_accuracy" : valid_acc}, epoch)

        writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch)

        if epoch == 0:
            best_loss = valid_loss

        else:
            if valid_loss < best_loss:
                print(f"Valid loss decreased. The minimum valid loss updated {best_loss:.4f} to {valid_loss:.4f}.")
                best_loss = valid_loss
                EARLY_STOP_PATIENCE = config["early_stop_patience"]
                torch.save(model.state_dict(), f"{save_path}/weights/best_{epoch}_{valid_acc}.pth")
            else:
                print(f"Valid loss did not decrease. best : {best_loss:.4f} | current : {valid_loss:.4f}")
                EARLY_STOP_PATIENCE -= 1
                if EARLY_STOP_PATIENCE == 0:
                    print("Early stopping patience is 0. Train stopped.")
                    break

    writer.close()
    torch.save(model.state_dict(), f"{save_path}/weights/last_{epoch}_{valid_acc}.pth")