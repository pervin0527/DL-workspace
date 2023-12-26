import os
import cv2
import torch
import pickle
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset

from data.augmentation import get_transform

def detection_collate(batch):
    bsize = len(batch)
    im_data, boxes, gt_classes, num_obj = zip(*batch)
    max_num_obj = max([x.item() for x in num_obj]) ##  배치 내에서 가장 많은 객체를 가진 이미지를 기준으로 다른 이미지들의 객체 수를 맞춘다.
    
    ## 모든 이미지의 바운딩 박스와 클래스 레이블을 저장하기 위한 것으로, 초기에는 0으로 채워진다.
    padded_boxes = torch.zeros((bsize, max_num_obj, 4))
    padded_classes = torch.zeros((bsize, max_num_obj,))

    for i in range(bsize):
        padded_boxes[i, :num_obj[i], :] = boxes[i]
        padded_classes[i, :num_obj[i]] = gt_classes[i]

    return torch.stack(im_data, 0), padded_boxes, padded_classes, torch.stack(num_obj, 0)

class VOCDataset(Dataset):
    def __init__(self, root_dir, image_sets, years, use_diff=False, img_size=416):
        super(VOCDataset, self).__init__()
        self.root_dir = root_dir
        self.image_sets = image_sets
        self.years = years
        self.use_diff = use_diff
        self.img_size = img_size

        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        
        self.images, self.annots = [], []
        self.load_data()


    def load_image_set_index(self, image_set, year):
        image_set_file = f"{self.root_dir}/VOC{year}/ImageSets/Main/{image_set}.txt"
        with open(image_set_file) as f:
            image_index = [f"{self.root_dir}/VOC{year}/JPEGImages/{x.strip()}.jpg" for x in f.readlines()]

        return image_index
    

    def load_pascal_annotation(self, index):
        filename = index.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")
        tree = ET.parse(filename)
        objs = tree.findall('object')

        if not self.use_diff:
            non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs

        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            cls = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        return {'boxes': boxes, 'gt_classes': gt_classes}


    def load_annotation_set_index(self, image_set, year, image_index):
        cache_file = f"{self.root_dir}/VOC{year}/{image_set}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                roidb = pickle.load(f)

            return roidb
        
        gt_roidb = [self.load_pascal_annotation(index) for index in image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        
        return gt_roidb


    def load_data(self):
        for year in self.years:
            for image_set in self.image_sets:
                image_index = self.load_image_set_index(image_set, year)
                annot_index = self.load_annotation_set_index(image_set, year, image_index)
                self.images.extend(image_index)
                self.annots.extend(annot_index)


    def __len__(self):
        
        return len(self.images)


    def __getitem__(self, idx):
        image = self.images[idx]
        annot = self.annots[idx]
        boxes, gt_classes = annot["boxes"], annot["gt_classes"]

        # image = Image.open(image)
        # image = image.convert('RGB')
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_info = torch.FloatTensor([image.shape[0], image.shape[1]])

        if "train" in  self.image_sets:
            transform = get_transform(is_train=True, img_size=self.img_size)
            transformed = transform(image=np.array(image), bboxes=boxes, labels=gt_classes)
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"])
            gt_classes = np.array(transformed["labels"])

            w, h = image.shape[:2]
            boxes[:, 0::2] = np.clip(boxes[:, 0::2] / w, 0.001, 0.999)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2] / h, 0.001, 0.999)

            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
            boxes = torch.from_numpy(boxes)
            gt_classes = torch.from_numpy(gt_classes)
            num_obj = torch.Tensor([boxes.size(0)]).long()
            
            return image, boxes, gt_classes, num_obj

        else:
            transform = get_transform(is_train=False, img_size=self.img_size)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255
            
            return image, im_info


if __name__ == "__main__":
    train_dataset = VOCDataset(root_dir="/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit", image_sets=['train', 'val'], years=['2007', '2012'], use_diff=False)
    print(len(train_dataset))

    data = train_dataset[0]
    image, boxes, classes = data[0], data[1], data[2]
    print(image.shape)