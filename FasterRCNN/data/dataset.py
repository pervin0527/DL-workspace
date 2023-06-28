import os
import cv2
import json
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from xml.etree import ElementTree as ET

class PascalVOCDataset(Dataset):
    def __init__(self, dir, set_name, use_dfficult=False, transform=None):
        self.dir = dir
        self.set_name = set_name
        self.transform = transform
        self.use_difficult = use_dfficult

        if set_name == "train":
            data_list_file = f"{self.dir}/ImageSets/Main/trainval.txt"
        else:
            data_list_file = f"{self.dir}/ImageSets/Main/{self.set_name}.txt"
        
        with open(data_list_file, "r") as f:
            file_list = f.readlines()
        
        self.file_list = [x.strip() for x in file_list]
        self.classes = self.get_classes(self.file_list)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_file_path = f"{self.dir}/JPEGImages/{file_name}.jpg"
        xml_file_path = f"{self.dir}/Annotations/{file_name}.xml"

        image = cv2.imread(image_file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, labels, area, is_difficult = self.read_xml(xml_file_path)
        labels = [self.classes.index(label) for label in labels]

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int32)
        area = torch.as_tensor(area, dtype=torch.float32)
        isdifficult = torch.as_tensor(is_difficult, dtype=torch.int32)
        

        target = {"boxes": boxes, "labels": labels, "area": area, "isdifficult" : isdifficult}

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
    
    def read_xml(self, path):
        annots = ET.parse(open(path, "r"))
        root = annots.getroot()

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        channels = int(size.find("depth").text)

        boxes, labels = [], []
        objects = root.findall("object")
        for obj in objects:
            is_difficult = int(obj.find("difficult").text)
            if self.use_difficult == False and is_difficult == 1:
                continue

            name = obj.find("name").text
            labels.append(name)

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width - 1, xmin + max(0, xmax - 1))
            ymax = min(height - 1, ymin + max(0, ymax - 1))
            area = (xmax - xmin) * (ymax - ymin)

            if area > 0 and xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        return boxes, labels, area, is_difficult
    
    def get_classes(self, files):
        totals = []
        for file in files:
            path = f"{self.dir}/Annotations/{file}.xml"
            _, labels, _, _ = self.read_xml(path)

            for label in labels:
                if label not in totals:
                    totals.append(label)
        
        totals.sort()
        totals.insert(0, "background")
        return totals


class COCODataset(Dataset):
    def __init__(self, dir, set_name, year, transforms=None):
        self.dir = dir
        self.year = year
        self.set_name = set_name
        self.transforms = transforms

        if set_name != "test":
            instance_path = f"{self.dir}/annotations/instances_{set_name}{year}.json"
            self.coco_api = COCO(instance_path)

            with open(instance_path) as f:
                ## info, licenses, images, annotations, categories
                self.dataset = json.load(f)
            
            cats = self.coco_api.loadCats(self.coco_api.getCatIds())
            self.classes = tuple(['__background__'] + [c['name'] for c in cats])
            self.num_classes = len(self.classes)

            self.class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
            self.class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats], self.coco_api.getCatIds())))
            self.coco_cat_id_to_class_ind = dict([(self.class_to_coco_cat_id[cls], self.class_to_ind[cls]) for cls in self.classes[1:]])

            
    def __len__(self):
        return len(self.dataset["images"])
    
    def __getitem__(self, idx):
        data = self.dataset["images"][idx]
        image_idx = data['id']
        file_name = f"{(str(image_idx).zfill(12))}.jpg"
        image_path = f"{self.dir}/{self.set_name}{self.year}/{file_name}"

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width, height = image.shape[:2]

        annIds = self.coco_api.getAnnIds(imgIds=image_idx, iscrowd=None)
        objs = self.coco_api.loadAnns(annIds)

        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)

        objs = valid_objs
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        iscrowd = []
        for ix, obj in enumerate(objs):
            cls = self.coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            iscrowd.append(int(obj["iscrowd"]))

        image_id = torch.tensor([image_idx])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        gt_classes = torch.as_tensor(gt_classes, dtype=torch.int32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": gt_classes, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    

if __name__ == "__main__":
    # test = COCODataset("/home/pervinco/Datasets/COCO", "train", "2017", None)
    # print(test[0])

    test = PascalVOCDataset("/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012", "train", False, None)
    print(test[0])