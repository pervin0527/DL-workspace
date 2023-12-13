import os
import torch

from tqdm import tqdm
from datetime import datetime

from torch import nn
from torch import optim
from torchsummary import summary
from torch.utils.data import DataLoader

from models.util import load_model

from data.augmentation import get_transform
from data.dataset import ClassificationDataset

from utils.graph import plot_and_save
from utils.train_param import read_train_params, save_train_params
from utils.lr_scheduler import LinearWarmupDecayScheduler


def eval(model, dataloader, criterion, device):
    model.eval()

    eval_loss, eval_accuracy = 0.0, 0.0
    with torch.no_grad():
        for (batch_images, batch_labels) in tqdm(dataloader, desc="Eval", leave=False):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            prediction = model(batch_images)
            loss = criterion(prediction, batch_labels)

            eval_loss += loss.item() * batch_images.size(0)
            
            _, predicted_classes = torch.max(prediction, 1)
            eval_accuracy += (predicted_classes == batch_labels).sum().item()
            
    eval_loss /= len(dataloader.dataset)
    eval_accuracy /= len(dataloader.dataset)

    return eval_loss, eval_accuracy


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    
    train_loss, train_accuracy = 0.0, 0.0
    for (batch_images, batch_labels) in tqdm(dataloader, desc="Train", leave=False):
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        prediction = model(batch_images)

        loss = criterion(prediction, batch_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_images.size(0)

        _, predicted_classes = torch.max(prediction, 1)
        train_accuracy += (predicted_classes == batch_labels).sum().item()

    train_loss /= len(dataloader.dataset)
    train_accuracy /= len(dataloader.dataset)

    return train_loss, train_accuracy


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_params = read_train_params("./config.yaml")

    train_transform = get_transform(is_train=True, img_size=train_params["img_size"])
    valid_transform = get_transform(is_train=False, img_size=train_params["img_size"])

    train_dataset = ClassificationDataset(data_dir=train_params["data_dir"], transform=train_transform, is_train=True)
    valid_dataset = ClassificationDataset(data_dir=train_params["data_dir"], transform=valid_transform, is_train=False)
    classes = train_dataset.get_classes()
    print(len(train_dataset), len(valid_dataset))

    save_dir = train_params["save_dir"]
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = f"{save_dir}/{current_time}"

    if not os.path.isdir(save_path):
        print(save_path)
        os.makedirs(f"{save_path}/weights")
        os.makedirs(f"{save_path}/plots")
        train_params["save_dir"] = save_path

    save_train_params(save_path, train_params)

    train_dataloader = DataLoader(train_dataset, batch_size=train_params["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_params["batch_size"])

    model = load_model(model_name=train_params["model_name"], num_classes=len(classes), init_weights=True, pretrained=train_params["pretrained"])
    summary(model, input_size=(3, 224, 224), device="cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=train_params["init_lr"],  weight_decay=train_params["weight_decay"])

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


        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device)
        print(f"\tTrain Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")

        valid_loss, valid_acc = eval(model, valid_dataloader, criterion, device)
        print(f"\tValid Loss : {valid_loss:.4f}, Valid Acc : {valid_acc:.4f}")

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

    plot_and_save(train_losses, 'Training Loss', 'Loss', f'{save_path}/plots/train_loss.png')
    plot_and_save(valid_losses, 'Validation Loss', 'Loss', f'{save_path}/plots/valid_loss.png')
    plot_and_save(learning_rates, 'Learning Rate', 'Learning Rate', f'{save_path}/plots/learning_rate.png')
    

if __name__ == "__main__":
    main()