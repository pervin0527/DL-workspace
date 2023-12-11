import os
import torch

from tqdm import tqdm

from torch import nn
from torch import optim
from torchsummary import summary
from torch.utils.data import DataLoader

from config import *
from models.util import load_model
from data.augmentation import get_transform
from data.dataset import ClassificationDataset


def eval(model, dataloader, criterion):
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



def train(model, dataloader, criterion, optimizer):
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
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    model = load_model(model_name=MODEL_NAME, num_classes=len(classes), init_weights=True, pretrained=PRETRAINED)
    summary(model, input_size=(3, 224, 224), device="cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE,  weight_decay=WEIGHT_DECAY)

    min_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f"\nEpoch : [{epoch + 1} | {EPOCHS}]")
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
        print(f"\tTrain Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")
        valid_loss, valid_acc = eval(model, valid_dataloader, criterion)
        print(f"\tValid Loss : {valid_loss:.4f}, Valid Acc : {valid_acc:.4f}")

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{SAVE_DIR}/ep_{epoch+1}_{valid_loss:.4f}.pth')
            print(f"Model saved at epoch {epoch+1} with validation loss {valid_loss:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_transform = get_transform(is_train=True, img_size=IMG_SIZE)
    valid_transform = get_transform(is_train=False, img_size=IMG_SIZE)

    train_dataset = ClassificationDataset(data_dir=DATA_DIR, transform=train_transform, is_train=True)
    valid_dataset = ClassificationDataset(data_dir=DATA_DIR, transform=valid_transform, is_train=False)
    print(len(train_dataset), len(valid_dataset))

    classes = train_dataset.get_classes()

    main()