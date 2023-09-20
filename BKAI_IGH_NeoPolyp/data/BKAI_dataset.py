import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from glob import glob
from torch.utils.data import Dataset


class BKAIDataset(Dataset):
    def __init__(self, base_dir, split, size=256, threshold=50):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.size = (size, size)
        self.threshold = threshold

        self.set_txt = f"{base_dir}/{split}.txt"
        with open(self.set_txt, "r") as f:
            file_list = f.readlines()

        self.file_list = [x.strip() for x in file_list]
        

    def __getitem__(self, index):
        file_name = self.file_list[index]
        image = cv2.imread(f"{self.base_dir}/train/{file_name}.jpeg")
        mask = cv2.imread(f"{self.base_dir}/train_gt/{file_name}.jpeg") 
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1)) ## H, W, C -> C, H, W
        image = image / 255.0

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, self.size)
        mask = self.convert_mask(mask)

        return image, mask
    

    def __len__(self):
        return len(self.file_list)


    def convert_mask(self, mask):
        label_transformed = np.full(mask.shape[:2], 0, dtype=np.uint8)

        red_mask = (mask[:, :, 0] > self.threshold) & (mask[:, :, 1] < 50) & (mask[:, :, 2] < 50)
        label_transformed[red_mask] = 1

        green_mask = (mask[:, :, 0] < 50) & (mask[:, :, 1] > self.threshold) & (mask[:, :, 2] < 50)
        label_transformed[green_mask] = 2

        return label_transformed