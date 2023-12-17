import torch

from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader

from loss import YoloLoss
from models.yolov1 import Yolov1
from data.dataset import VOCDataset

from utils.train_param import read_train_params, save_train_params
from utils.detection_utils import mean_average_precision, get_bboxes


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train(dataloader, model, optimizer, loss_func, device):
    model.train()

    train_loss = 0.0
    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)

        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    return train_loss


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_params = read_train_params("./config.yaml")

    model = Yolov1(grid_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_params["init_lr"], weight_decay=train_params["weight_decay"])
    loss_func = YoloLoss()

    train_dataset = VOCDataset("data/train.csv", transform=transform)
    test_dataset = VOCDataset("data/test.csv", transform=transform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_params["batch_size"], shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=train_params["batch_size"])

    total_epochs = train_params["epochs"]
    for epoch in range(total_epochs):
        print(f"\nEpoch : [{epoch + 1} | {total_epochs}]")

        train_loss = train(train_dataloader, model, optimizer, loss_func, device)

        pred_boxes, target_boxes = get_bboxes(train_dataloader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")
        print(f"\nTrain Loss : {train_loss}, Train mAP : {mean_avg_prec}")



if __name__ == "__main__":
    main()
