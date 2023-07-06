import time
import torch
import numpy as np
import torch.utils.tensorboard
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from config import train_cfg
from model import YoloV3
from dataset import DetectionDataset
from utils import init_weights_normal, xywh2xyxy, ap_per_class, non_max_suppression, get_batch_statistics

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(DEVICE, torch.cuda.get_device_name(0))
    else:
        DEVICE = torch.device("cpu")
        print(DEVICE)

    ## Tensorboard Writer
    writer = torch.utils.tensorboard.SummaryWriter(f"{train_cfg.save_dir}/logs")

    ## Training Dataset
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DetectionDataset(data_dir=train_cfg.data_dir, set_name="train", annot_type="voc", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    print("Train Dataset Loaded.")
    classes = train_dataset.classes

    ## Validation Dataset
    valid_transform = transforms.Compose([transforms.ToTensor()])
    valid_dataset = DetectionDataset(data_dir=train_cfg.data_dir, set_name="valid", annot_type="voc", transform=valid_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_cfg.batch_size)
    print("Valid Dataset Loaded.")

    ## Build Model
    model = YoloV3(image_size=train_cfg.train_img_size, num_classes=len(classes)).to(DEVICE)
    model.apply(init_weights_normal)
    dummy = torch.zeros((1, 3, train_cfg.train_img_size, train_cfg.train_img_size)).to(DEVICE)
    writer.add_graph(model, dummy)
    writer.close()

    ## Training
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg.step_size, gamma=train_cfg.gamma)

    best_loss = 0
    for epoch in range(train_cfg.epochs):
        model.train()
        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1} / {train_cfg.epochs}', unit='step')
        
        for idx, (file_name, images, targets) in enumerate(train_dataloader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            loss, outputs = model(images, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss" : loss.item()})
            pbar.update(1)

        # pbar.close()
            break
        break