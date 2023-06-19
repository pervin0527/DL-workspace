import torch
import torchvision.transforms as transforms

from coco_loader import CocoDataset
from torch.utils.data import DataLoader
from config import Confing

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    cfg = Confing()

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CocoDataset(cfg.DATA_PATH, "train", "2017", train_transform)
    valid_dataset = CocoDataset(cfg.DATA_PATH, "valid", "2017", valid_transform)

    train_dataloader = DataLoader(train_dataset, cfg.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, cfg.BATCH_SIZE)