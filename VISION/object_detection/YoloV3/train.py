import os
import torch

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from models.yolov3 import YoloV3
from data.dataset import YoloDataset
from utils.util import read_yaml, save_yaml, make_log_dir

def train(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0
    for paths, images, targets in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss, output = model(images, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= len(dataloader.dataset)

    return train_loss


def main():
    now = datetime.now().strftime('%y%m%d_%H%M%S')
    print(now)

    train_path = cfg['train']
    valid_path = cfg['valid']
    class_names = cfg['names']
    print(f"{train_path}\n{valid_path}\n{class_names}")

    train_dataset = YoloDataset(train_path, cfg["img_size"], augment=True, multiscale=True)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=num_workers, collate_fn=train_dataset.collate_fn)

    model = YoloV3(anchors=cfg["anchors"], img_size=cfg["img_size"], num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    epochs = cfg["epochs"]
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        print(f"{train_loss:.4f}")
    

if __name__ == "__main__":
    num_workers = os.cpu_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = read_yaml("./config.yaml")

    main()