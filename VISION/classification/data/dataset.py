import os
import cv2
import math
import random
import pandas as pd

from torch.utils.data import Dataset


class PlantPathologyDataset(Dataset):
    classes = ["healthy",
               "complex",
               "frog_eye_leaf_spot",
               "frog_eye_leaf_spot complex",
               "powdery_mildew",
               "powdery_mildew complex",
               "rust",
               "rust complex",
               "rust frog_eye_leaf_spot",
               "scab",
               "scab frog_eye_leaf_spot",
               "scab frog_eye_leaf_spot complex"]
    
    def __init__(self,dataset, image_size, is_train):
        self.dataset = dataset
        
        if is_train:
            random.shuffle(self.dataset)

        self.image_size = (image_size, image_size)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx].split(",")
        image_path, label = data[0], self.classes.index(data[1:][0])

        return image_path, label