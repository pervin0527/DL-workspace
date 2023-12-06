import torch
import importlib

from tqdm import tqdm
from torch import nn
from torch import optim
from torchsummary import summary
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import *
from models.util import load_model
from data.data_util import get_datasets
from data.augmentation import get_transform
from data.dataset import PlantPathologyDataset

def eval(model, dataloader, criterion):
    model.eval()

    eval_loss, correct_preds, total_preds = 0, 0, 0
    with torch.no_grad():
        for (batch_images, batch_labels) in tqdm(dataloader, desc="Eval", leave=False):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            prediction = model(batch_images)
            prediction = F.sigmoid(prediction)
            loss = criterion(prediction, batch_labels)

            eval_loss = loss.item * batch_images.size(0)
            preds = prediction > 0.5
            correct_preds += (preds == batch_labels.byte()).all(1).sum().item()
            total_preds += batch_labels.size(0)

    eval_loss = eval_loss / len(dataloader.dataset)
    eval_acc = correct_preds / total_preds

    return eval_loss, eval_acc


def train(model, dataloader, criterion, optimizer):
    model.train()
    
    train_loss, correct_preds, total_preds = 0, 0, 0
    for (batch_images, batch_labels) in tqdm(dataloader, desc="Train", leave=False):
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        prediction = model(batch_images)
        prediction = F.sigmoid(prediction)

        loss = criterion(prediction, batch_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_images.size(0)
        preds = prediction > 0.5
        correct_preds += (preds == batch_labels.byte()).all(1).sum().item()
        total_preds += batch_labels.size(0)

    train_loss = train_loss / len(dataloader.dataset)
    train_acc = correct_preds / total_preds

    return train_loss, train_acc


def main():
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    model = load_model(model_name=MODEL_NAME, num_classes=len(classes))
    summary(model, input_size=(3, 224, 224), device="cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(params=model.parameters(),
                           lr=LEARNING_RATE, 
                           weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        print(f"Epoch : [{epoch + 1} | {EPOCHS}]")
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
        print(f"\tTrain Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")
        valid_loss, valid_acc = eval(model, valid_dataloader, criterion)
        print(f"\tValid Loss : {valid_loss:.4f}, Valid Acc : {valid_acc:.4f}")



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    classes = PlantPathologyDataset.get_classes()
    train_dataset, valid_dataset = get_datasets(DATA_DIR, valid_ratio=VALID_RATIO)

    train_transform = get_transform(is_train=True, img_size=IMG_SIZE)
    valid_transform = get_transform(is_train=False, img_size=IMG_SIZE)

    train_dataset = PlantPathologyDataset(train_dataset, image_size=IMG_SIZE, transform=train_transform, is_train=True)
    valid_dataset = PlantPathologyDataset(valid_dataset, image_size=IMG_SIZE, transform=valid_transform, is_train=False)

    main()