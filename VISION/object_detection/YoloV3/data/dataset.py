import os
import cv2

from torch.utils.data import Dataset

from data.util import pad_to_square


class YoloDataset(Dataset):
    def __init__(self, file_list_path, img_size, augment, multiscale, normalized_labels):
        self.img_size = img_size
        self.augment = augment
        self.max_objects = 100
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.batch_count = 0

        with open(file_list_path, 'r') as f:
            self.image_files = f.readlines()

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('JPEGImages', 'labels') for path in self.image_files]


    def __getitem__(self, idx):
        img_path = self.image_files[idx].rstrip()
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        label_path = self.label_files[idx].rstrip()

        targets = None
        if os.path.exists(label_path):