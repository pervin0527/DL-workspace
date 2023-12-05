from torch.utils.data import DataLoader

from data.data_util import get_datasets
from data.augmentation import get_transform
from data.dataset import PlantPathologyDataset

if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/plant-pathology-2021"
    batch_size = 32
    valid_ratio = 0.2
    img_size = 512
    train_dataset, valid_dataset = get_datasets(data_dir, valid_ratio=valid_ratio)

    train_transform = get_transform(is_train=True, img_size=img_size)
    valid_transform = get_transform(is_train=False, img_size=img_size)

    train_dataset = PlantPathologyDataset(train_dataset, image_size=img_size, transform=train_transform, is_train=True)
    valid_dataset = PlantPathologyDataset(valid_dataset, image_size=img_size, transform=valid_transform, is_train=False)

    for idx, data in enumerate(train_dataset):
        if idx == 10:
            break

        image, one_hot_labels = data
        print(image.shape, one_hot_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)