import os
import torch
import torch.utils.tensorboard
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from model import YoloV3
from dataset import DetectionDataset
from eval import evaluate
from config import train_cfg, test_cfg
from utils import init_weights_normal

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(DEVICE, torch.cuda.get_device_name(0))
    else:
        DEVICE = torch.device("cpu")
        print(DEVICE)

    ## Tensorboard Writer
    writer = torch.utils.tensorboard.SummaryWriter(f"{train_cfg.save_dir}/logs")
    num_worker = min([os.cpu_count(), train_cfg.batch_size if train_cfg.batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(num_worker))

    ## Training Dataset
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DetectionDataset(data_dir=train_cfg.data_dir, set_name="train", annot_type="voc", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=num_worker, collate_fn=train_dataset.collate_fn)
    print("Train Dataset Loaded.")
    classes = train_dataset.classes

    ## Validation Dataset
    valid_transform = transforms.Compose([transforms.ToTensor()])
    valid_dataset = DetectionDataset(data_dir=train_cfg.data_dir, set_name="valid", annot_type="voc", transform=valid_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=test_cfg.batch_size, num_workers=num_worker, collate_fn=valid_dataset.collate_fn)
    print("Valid Dataset Loaded. \n")

    ## Build Model
    model = YoloV3(image_size=train_cfg.img_size, num_classes=len(classes)).to(DEVICE)
    model.apply(init_weights_normal)

    ## Training
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    best_ap = 0
    lr_patience = train_cfg.lr_patience
    for epoch in range(train_cfg.epochs):
        model.train()
        train_loss = 0
        train_bbox_loss, train_conf_loss, train_cls_loss = 0, 0, 0
        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1} / {train_cfg.epochs}', unit='step')
        for batch_idx, (file_name, images, targets) in enumerate(train_dataloader):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            loss, outputs = model(images, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            for i, yolo_layer in enumerate(model.yolo_layers):
                train_bbox_loss += yolo_layer.metrics["loss_bbox"] * images.size(0)
                train_conf_loss += yolo_layer.metrics["loss_conf"] * images.size(0)
                train_cls_loss += yolo_layer.metrics["loss_cls"] * images.size(0)

            pbar.set_postfix({"Train_Loss" : loss.item()})
            pbar.update(1) 

        pbar.close()
        train_loss /= len(train_dataloader.dataset)
        train_bbox_loss /= len(train_dataloader.dataset)
        train_conf_loss /= len(train_dataloader.dataset)
        train_cls_loss /= len(train_dataloader.dataset)
        print(f"Train Loss: {train_loss:.4f}, train_box_loss : {train_bbox_loss:.4f}, train_conf_loss : {train_conf_loss:.4f}, train_cls_loss : {train_cls_loss:.4f}")
            
        precision, recall, AP, f1, _, _, _ = evaluate(model, valid_dataloader, DEVICE)

        if epoch == 0:
            best_ap = AP.mean()
        else:
            if AP.mean() <= best_ap:
                print(f"Average Precision Increased.")
                best_ap = AP.mean()
                torch.save(model.state_dict(), f"{train_cfg.save_dir}/{epoch}.pth")
            else:
                lr_patience -= 1
                print("Average Precision Not Increase.")

                if lr_patience == 0:
                    lr_patience = train_cfg.lr_patience
                    current_lr = optimizer.param_groups[0]["lr"]
                    new_lr = current_lr * train_cfg.lr_factor
                    optimizer.param_groups[0]['lr'] = new_lr
                    print(f"Learning Rate Changed. {current_lr:.4f} to {new_lr:.4f}")

    torch.save(model.state_dict(), f"{train_cfg.save_dir}/final.pth")