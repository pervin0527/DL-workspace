import cv2
import torch

from glob import glob
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

class MyDataset(Dataset):
    def __init__(self, data_dir, set_name, img_size, transform=None):
        self.data_dir = f"{data_dir}/{set_name}"
        self.transform = transform
        self.img_size = (img_size, img_size)

        class_str = sorted(glob(f"{self.data_dir}/*"))
        self.class_str = [x.split('/')[-1] for x in class_str]
        
        self.image_files = sorted(glob(f"{self.data_dir}/*/*.jpg"))
        
    def __len__(self):
        return len(self.image_files)
    
    def get_classes(self):
        return self.class_str
    
    def __getitem__(self, index):
        file = self.image_files[index]
        
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        
        if self.transform != None:
            image = self.transform(image)

        label = self.class_str.index(file.split('/')[-2])
        label = one_hot(torch.tensor(label), len(self.class_str)).type(torch.float32)

        return image, label