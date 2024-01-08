import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset

from data.augmentation import get_transform
from data.util import resize_image_and_boxes


class YoloDataset(Dataset):
    def __init__(self, file_list_path, img_size, augment=None, multiscale=True):
        self.batch_count = 0
        self.max_objects = 100
        self.img_size = img_size
        self.augment = get_transform() if augment else None
        self.multiscale = multiscale

        with open(file_list_path, 'r') as f:
            self.image_files = f.readlines()

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('JPEGImages', 'labels') for path in self.image_files]


    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        img_path = self.image_files[idx].rstrip()
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annot_path = self.label_files[idx].rstrip()
        if os.path.exists(annot_path):
            annot = np.loadtxt(annot_path).reshape(-1, 5)
            if annot.size > 0:
                class_ids = annot[:, 0].reshape(-1, 1).astype(np.int64)
                boxes = annot[:, 1:]

                if self.augment is not None:
                    transformed = self.augment(image=image, bboxes=boxes, labels=class_ids)
                    image = transformed["image"]
                    boxes = np.array(transformed["bboxes"])
                    class_ids = np.array(transformed["labels"])

                    # 데이터 차원 확인 및 수정
                    if class_ids.ndim == 1:
                        class_ids = class_ids.reshape(-1, 1)
                    if boxes.ndim == 1:
                        boxes = boxes.reshape(-1, 1)

                # 빈 배열 처리
                if len(class_ids) == 0 or len(boxes) == 0:
                    boxes = np.zeros((0, 5))
                else:
                    boxes = np.concatenate((class_ids, boxes), axis=1)

            else:
                # 객체가 없는 경우 빈 boxes 배열 처리
                boxes = np.zeros((0, 5))

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = torch.from_numpy(boxes)
        
        return img_path, torch.tensor(image, dtype=torch.float32,), targets
    

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # 멀티스케일 조정
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(320, 608 + 1, 32))

        # 이미지와 타겟의 크기 조정
        resized_images = []
        resized_targets = []
        for i, (image, target) in enumerate(zip(images, targets)):
            if target is not None:
                # 이미지와 바운딩 박스 크기 조정
                image, boxes = resize_image_and_boxes(image.numpy(), target[:, 2:], new_size=self.img_size)
                image = np.transpose(image, axes=(2, 0, 1))
                resized_images.append(torch.from_numpy(image))

                # 타겟에 배치 인덱스 추가
                target[:, 0] = i  # 첫 번째 열에 배치 인덱스 할당
                target[:, 2:] = torch.from_numpy(boxes)  # 바운딩 박스 좌표 업데이트
                resized_targets.append(target)
            else:
                # 이미지만 크기 조정 (타겟이 없는 경우)
                image = resize_image_and_boxes(image, None, new_size=self.img_size)[0]
                image = np.transpose(image, axes=(2, 0, 1))
                resized_images.append(torch.from_numpy(image))

        # 타겟 텐서 생성
        if len(resized_targets) > 0:
            targets = torch.cat(resized_targets, 0)
        else:
            targets = None

        self.batch_count += 1
        
        return paths, torch.stack(resized_images), targets