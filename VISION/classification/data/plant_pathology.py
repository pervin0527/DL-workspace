import cv2
import torch
import random

from torch.utils.data import Dataset


class PlantPathologyDataset(Dataset):
    ## healthy, complex, frog_eye_leaf_spot, powdery_mildew, rust, scab ---> 6 classes
    classes = ["healthy", "complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]
    
    def __init__(self,dataset, image_size, is_train, transform=None):
        self.dataset = dataset
        self.img_size = (image_size, image_size)
        self.transform = transform
        
        if is_train:
            random.shuffle(self.dataset)

        self.image_size = (image_size, image_size)

    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def get_classes():
        return PlantPathologyDataset.classes
    
    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # h, w, c = image.shape
        # image = torch.tensor(image).view(c, h, w)

        return image
    
    def str_to_idx(self, labels):
        label = [self.classes.index(label) for label in labels]

        return label
    
    def one_hot_encoding(self, labels):
        one_hot_label = torch.zeros(len(self.classes), dtype=torch.float32)
        # src 텐서를 labels의 길이와 같게 설정
        src = torch.ones(len(labels), dtype=torch.float32)
        one_hot_label.scatter_(dim=0, index=torch.tensor(labels, dtype=torch.int64), src=src)

        return one_hot_label


    def __getitem__(self, idx):
        data = self.dataset[idx].split(",")
        image_path, labels = data[0], data[1:][0].split()
        
        image = self.read_image(image_path)
        
        labels = self.str_to_idx(labels)
        one_hot_label = self.one_hot_encoding(labels)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, one_hot_label