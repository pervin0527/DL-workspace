from data.dataset import PlantPathologyDataset
from data.data_util import get_datasets

data_dir = "/home/pervinco/Datasets/plant-pathology-2021"
valid_ratio = 0.2
train_dataset, valid_dataset = get_datasets(data_dir, valid_ratio=valid_ratio)


train_dataset = PlantPathologyDataset(train_dataset, image_size=512, is_train=True)

for idx, data in enumerate(train_dataset):
    if idx == 10:
        break
    print(data)