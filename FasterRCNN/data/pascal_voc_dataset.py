import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

class PascalVocDataset(Dataset):
    def __init__(self, data_dir, dataset_type, transforms=None):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.transforms = transforms

        self.list_file_path = f"{self.data_dir}/ImageSets/Main/{self.dataset_type}.txt"
        with open(self.list_file_path, 'r') as f:
            self.file_list = f.readlines()
        
        self.file_list = [x.strip() for x in self.file_list]
        self.classes = self.get_classes()

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        data = self.file_list[index]
        image_path = f"{self.data_dir}/JPEGImages/{data}.jpg"
        xml_path = f"{self.data_dir}/Annotations/{data}.xml"

        image = cv2.imread(image_path)
        height, width, channel = image.shape

        annotation = ET.parse(xml_path)
        bboxes, labels, iscrowd = [], [], []
        for obj in annotation.findall("object"):
            iscrowd.append(int(obj.find('difficult').text))
            name = obj.find("name").text.lower().strip()
            labels.append(self.classes.index(name))

            bndbox = obj.find("bndbox")
            x1 = np.max((0, int(bndbox.find("xmin").text)))
            y1 = np.max((0, int(bndbox.find("ymin").text)))
            x2 = np.min((width - 1, x1 + np.max((0, int(bndbox.find("xmax").text) - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, int(bndbox.find("ymax").text) - 1))))

            if (x2 - x1) * (y2 - y1) > 0 and x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])

        image_id = torch.tensor(int(''.join(data.split('_'))))
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int32)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target = {"boxes": bboxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_classes(self):
        classes = set()
        for file in self.file_list:
            xml_path = f"{self.data_dir}/Annotations/{file}.xml"
            annotation = ET.parse(xml_path)
            
            for obj in annotation.findall("object"):
                classes.add(obj.find("name").text.lower().strip())
        classes = sorted(list(classes))
        classes.insert(0, "background")
        
        return classes
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    
if __name__ == "__main__":
    dataset = PascalVocDataset("/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012", "trainval", None)
    print(dataset.classes)

    for data in dataset:
        print(data)
        break