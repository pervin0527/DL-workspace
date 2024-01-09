import os
import cv2
import numpy as np
import albumentations as A
import xml.etree.ElementTree as ET

from data.augmentation import get_transform

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    # items[1] = torch.tensor(items[1])

    return items


class VOCDataset(Dataset):
    def __init__(self, root_path="/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit", year="2007", mode="train", image_size=448, is_training=True):
        self.img_size = image_size
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train','tvmonitor']

        if (mode in ["train", "val", "trainval", "test"] and year == "2007") or (mode in ["train", "val", "trainval"] and year == "2012"):
            self.data_path = os.path.join(root_path, "VOC{}".format(year)) ## "/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit" + "VOC2007" or "VOC2012"

        ## "/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit/VOC2012/ImageSets/Main/train.txt"
        id_list_path = os.path.join(self.data_path, "ImageSets/Main/{}.txt".format(mode))

        self.ids = [id.strip() for id in open(id_list_path)] ## 파일이름.jpg 리스트.
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.ids)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        id = self.ids[item]
        image_path = os.path.join(self.data_path, "JPEGImages", "{}.jpg".format(id))
        image_xml_path = os.path.join(self.data_path, "Annotations", "{}.xml".format(id))
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bboxes, labels = [], []
        annot = ET.parse(image_xml_path)
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())

            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            
        # transform = A.Compose([A.Resize(self.img_size, self.img_size)], bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]))
        transform = get_transform(self.is_training, self.image_size)
        transformed = transform(image=image, bboxes=bboxes)

        image = transformed["image"]
        boxes = transformed["bboxes"]
        objects = [[box[0], box[2], box[1], box[3], label] for box, label in zip(boxes, labels)]

        # return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
        return image , np.array(objects, dtype=np.float32)
    

if __name__ == "__main__":
    train_dataset = VOCDataset(image_size=416)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

    for data in train_dataloader:
        images, labels = data

        print(images.shape)
        print(len(labels))

        for label in labels:
            print(label.shape)

        break