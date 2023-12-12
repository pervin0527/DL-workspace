import yaml
import torch

from torch import nn, optim
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader import MyDataset
from models.model import MobileNetV1, MobileNetV2


def train(model, dataloader, loss_func, optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    
    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)
        
        Y_PRED = model(X)
        loss = loss_func(Y_PRED, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

        _, Y_PRED = torch.max(Y_PRED, 1)
        train_acc += (Y_PRED == Y).sum().item()

    train_loss /= len(dataloader.dataset)
    train_acc = train_acc / len(dataloader.dataset)

    return train_loss, train_acc


def valid(model, dataloader, loss_func):
    model.eval()
    valid_acc = 0
    valid_loss = 0
    
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    valid_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    ## Length of Dataset
    train_dataset = MyDataset(data_dir=config["data_dir"], set_name=config["train_set"], img_size=config["img_size"], transform=train_transform)
    valid_dataset = MyDataset(data_dir=config["data_dir"], set_name=config["valid_set"], img_size=config["img_size"], transform=valid_transform)

    ## Num of batches = len(dataset) / batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"])

    classes = train_dataset.get_classes()
    model = MobileNetV2(num_classes=len(classes))
    model.to(device)
    summary(model, (3, config["img_size"], config["img_size"]), device=device.type)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    epochs = config["epochs"]
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_dataloader, loss_func, optimizer)
        valid_loss, valid_acc = valid(model, valid_dataloader, loss_func)

        print(f"\nEPOCH[{epoch+1} | {epochs}]")
        print(f"Train Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")
        print(f"Valid Loss : {valid_loss:.4f}, Valid Acc : {valid_acc:.4f}")