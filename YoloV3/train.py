import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import DetectionDataset

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size = 1
    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012"

    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DetectionDataset(data_dir=data_dir, set_name="train", annot_type="voc", transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_transform = transforms.Compose([transforms.ToTensor()])
    valid_dataset = DetectionDataset(data_dir=data_dir, set_name="valid", annot_type="voc", transform=valid_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    for i in train_dataloader:
        print(i)
        break