import cv2
import json
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

class CoCoDataset(Dataset):
    def __init__(self, data_dir, dataset_type, year, transforms=None):
        self.data_dir = data_dir
        self.year = year
        self.dataset_type = dataset_type
        self.transforms = transforms

        self.json_file_path = f"{self.data_dir}/annotations/instances_{self.dataset_type}{self.year}.json"
        self.COCO = COCO(self.json_file_path)

        cats = self.COCO.loadCats(self.COCO.getCatIds())
        self.classes = tuple(['__background__'] + [c['name'] for c in cats])

        self.num_classes = len(self.classes)
        self.class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self.class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats], self.COCO.getCatIds())))
        self.coco_cat_id_to_class_ind = dict([(self.class_to_coco_cat_id[cls], self.class_to_ind[cls]) for cls in self.classes[1:]])

        with open(self.json_file_path) as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset['images'])

    def __getitem__(self, index):
        data = self.dataset['images'][index]
        
        image_id = data['id']
        file_name = f"{str(image_id).zfill(12)}.jpg"
        image_path = f"{self.data_dir}/{self.dataset_type}{self.year}/{file_name}"
        image = cv2.imread(image_path)
        height, width, channel = image.shape

        annIds = self.COCO.getAnnIds(imgIds=image_id, iscrowd=None)
        objects = self.COCO.loadAnns(annIds)

        confirmed_objects = []
        for obj in objects:
            x1 = np.max((0, obj['bbox'][0])) ## 0보다 작은 box 좌표일 경우 포함하지 않는다.
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1)))) ## width 보다 큰 box 좌표도 포함하지 않음.
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))

            if obj['area'] > 0 and x2 > x1 and y2 > y1: ## bounding box가 정상적인 형태를 가질 때
                obj['clean_bbox'] = [x1, y1, x2, y2]
                confirmed_objects.append(obj)

        num_objects = len(confirmed_objects)
        boxes = np.zeros((num_objects, 4), dtype=np.float32)
        labels = np.zeros((num_objects), dtype=np.int32)

        iscrowd = []
        for i, obj in enumerate(confirmed_objects):
            cls = self.coco_cat_id_to_class_ind[obj['category_id']]
            boxes[i, :] = obj['clean_bbox']
            labels[i] = cls
            iscrowd.append(int(obj["iscrowd"]))

        image_id = torch.tensor([image_id]) 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # @property
    # def class_to_coco_cat_id(self):
    #     return self.class_to_coco_cat_id

        
if __name__ == "__main__":
    train_dataset = CoCoDataset("/home/pervinco/Datasets/COCO", "train", "2017", None)
    print(len(train_dataset))

    for data in train_dataset:
        print(data)

        break