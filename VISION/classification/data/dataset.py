import os
import cv2

from glob import glob
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, data_dir, is_train, transform):
        if is_train:
            self.data_dir = f"{data_dir}/train"
        else:
            self.data_dir = f"{data_dir}/valid"

        self.transform = transform
        self.classes = os.listdir(self.data_dir)
        self.total_files = sorted(glob(f"{self.data_dir}/*/*.jpg"))

    def __len__(self):
        return len(self.total_files)
    
    def get_classes(self):
        return self.classes
    
    def read_image(self, file_path):
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    
    def str_to_idx(self, label):
        label = self.classes.index(label)

        return label
    
    def __getitem__(self, idx):
        file_path = self.total_files[idx]
        image = self.read_image(file_path)
        label = self.str_to_idx(file_path.split('/')[-2])

        if self.transform != None:
            transform_data = self.transform(image=image)
            image = transform_data["image"]

        return image, label