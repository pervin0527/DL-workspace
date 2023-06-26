import cv2
import numpy as np
import xml.etree.ElementTree as ET

class PascalVoc:
    def __init__(self, data_dir, split="trainval", use_difficult=False):
        txt_file = f"{data_dir}/ImageSets/Main/{split}.txt"
        
        with open(txt_file, "r") as f:
            contents = f.readlines()
        
        self.ids = [x.strip() for x in contents]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                        'bus', 'car', 'cat', 'chair', 'cow', 
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        annotation = ET.parse(f"{self.data_dir}/Annotations/{id}.xml")

        bboxes, labels, difficult = [], [], []
        for obj in annotation.findall('object'):
            if not self.use_difficult and int(obj.find("difficult").text) == 1:
                continue
            difficult.append(int(obj.find("difficult").text))

            bndbox = obj.find("bndbox")
            bboxes.append([int(bndbox.find(tag).text) for tag in ("ymin", "xmin", "ymax", "xmax")])
            
            name = obj.find("name").text.lower().strip()
            labels.append(self.classes.index(name))

        bboxes = np.stack(bboxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool_).astype(np.uint8)

        image = cv2.imread(f"{self.data_dir}/JPEGImages/{id}.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1)) ## (C, H, W)
        
        return image, bboxes, labels, difficult


if __name__ == "__main__":
    test = PascalVoc("/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012")
    print(test[0])