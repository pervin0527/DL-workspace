import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from configs.train_config import train_cfg
from utils.data_utils import CocoDataset
from utils.train_utils import build_model

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = CocoDataset(train_cfg.DATA_PATH, "train", "2017", train_transform)
    valid_dataset = CocoDataset(train_cfg.DATA_PATH, "val", "2017", valid_transform)

    train_dataloader = DataLoader(train_dataset, train_cfg.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, train_cfg.BATCH_SIZE)

    build_model(train_dataset.num_classes)