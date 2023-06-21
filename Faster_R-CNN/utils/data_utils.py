import os
import json
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(self, dataset_path, set_name, year, transform=None):
        self.dataset_path = dataset_path
        self.set_name = set_name
        self.year = year
        self.transform = transform
        self.get_annotation_data()
        self.num_classes = 0


    def get_annotation_data(self):
        if self.set_name != "test":
            self.annot_file = f"{self.dataset_path}/annotations/instances_{self.set_name}{self.year}.json"
        
        with open(self.annot_file) as file:
            self.annotation_data = json.load(file)
        
        self.COCO = COCO(self.annot_file)
        cats = self.COCO.loadCats(self.COCO.getCatIds())
        self.classes = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self.class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats], self.COCO.getCatIds())))
        self.coco_cat_id_to_class_ind = dict([(self.class_to_coco_cat_id[cls], self.class_to_ind[cls]) for cls in self.classes[1:]])


    def __len__(self):
        return len(self.annotation_data['images'])
    

    def __getitem__(self, idx):
        data = self.annotation_data["images"][idx]
        img_id = data["id"]
        img_file_name = str(img_id).zfill(12) + ".jpg"
        image_path = f"{self.dataset_path}/{self.set_name}{self.year}/{img_file_name}"
        
        print(image_path)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            width, height = data["width"], data["height"]
            annIds = self.COCO.getAnnIds(imgIds=img_id, iscrowd=None)
            objs = self.COCO.loadAnns(annIds)

            num_objs = len(objs)
            bboxes = []
            for obj in objs:
                x1 = np.max((0, obj['bbox'][0]))
                y1 = np.max((0, obj['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
                if obj['area'] > 0 and x2 > x1 and y2 > y1:
                    obj['clean_bbox'] = [x1, y1, x2, y2]
                    bboxes.append(obj)
            
            boxes = np.zeros((num_objs, 4), dtype=np.float32)
            gt_classes = np.zeros((num_objs), dtype=np.int32)

            iscrowd = []
            for ix, obj in enumerate(objs):
                cls = self.coco_cat_id_to_class_ind[obj['category_id']]
                boxes[ix, :] = obj['clean_bbox']
                gt_classes[ix] = cls
                iscrowd.append(int(obj["iscrowd"]))

            image_id = torch.tensor([img_id])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            gt_classes = torch.as_tensor(gt_classes, dtype=torch.int32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int32)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target = {"boxes": boxes, "labels": gt_classes, "image_id": image_id, "area": area, "iscrowd": iscrowd}

            if self.transform is not None:
                image, target = self.transform(image, target)

            return image, target


if __name__ == "__main__":
    test = CocoDataset("/home/pervinco/Datasets/COCO", "train", "2017")
    print(test[0])