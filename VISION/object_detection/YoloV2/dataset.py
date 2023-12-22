import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, root_path="/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit", year="2007", mode="train", image_size=448, is_training = True):
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
        
        annot = ET.parse(image_xml_path)
        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
    

if __name__ == "__main__":
    dataset = VOCDataset(image_size=416)
    sample_data = dataset[0]
    image, label = sample_data[0], sample_data[1]
    print(image.shape, label.shape)