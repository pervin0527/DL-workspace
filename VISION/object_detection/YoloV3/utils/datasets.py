import os
import cv2
import glob
import random
import numpy as np
import albumentations as A

import torch
import torch.utils.data
import torchvision.transforms
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def pad_to_square(image, pad_value=0):
    _, h, w = image.shape

    # 너비와 높이의 차
    difference = abs(h - w)

    # (top, bottom) padding or (left, right) padding
    if h <= w:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [0, 0, top, bottom]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [left, right, 0, 0]

    # Add padding
    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad


def resize(image, size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, folder_path, image_size):
        self.image_files = sorted(glob.glob("{}/*.*".format(folder_path)))
        self.image_size = image_size

    def __getitem__(self, index):
        image_path = self.image_files[index]

        # Extract image as PyTorch tensor
        image = torchvision.transforms.ToTensor()(Image.open(image_path).convert('RGB'))

        # Pad to square resolution
        image, _ = pad_to_square(image)

        # Resize
        image = resize(image, self.image_size)
        return image_path, image

    def __len__(self):
        return len(self.image_files)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, list_path: str, image_size: int, augment: bool, multiscale: bool, normalized_labels=True):
        with open(list_path, 'r') as file:
            self.image_files = file.readlines()

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('JPEGImages', 'labels') for path in self.image_files]
        self.image_size = image_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.batch_count = 0

    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.image_files[index].rstrip()

        # Apply augmentations
        if self.augment:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1),
                torchvision.transforms.ToTensor()
            ])
        else:
            transforms = torchvision.transforms.ToTensor()

        # Extract image as PyTorch tensor
        image = transforms(Image.open(image_path).convert('RGB'))

        _, h, w = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Pad to square resolution
        image, pad = pad_to_square(image)
        _, padded_h, padded_w = image.shape

        # 2. Label
        # -----------------------------------------------------------------------------------
        label_path = self.label_files[index].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)

        return image_path, image, targets

    def __len__(self):
        return len(self.image_files)

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Selects new image size every 10 batches
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(320, 608 + 1, 32))

        # Resize images to input shape
        images = torch.stack([resize(image, self.image_size) for image in images])
        self.batch_count += 1

        return paths, images, targets


def get_transform():
    transform = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
        ], p=1),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),

        A.OneOf([
            A.RandomRotate90(p=0.35),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.35),
            A.PadIfNeeded(min_height=608, min_width=608, border_mode=0, p=0.3)
        ], p=1),

        A.OneOf([
            A.Blur(p=0.5), 
            A.GaussianBlur(p=0.5), 
            A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=0.5)
        ], p=0.5),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
        ], p=0.5),

        # A.OneOf([
        #     A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.5),
        #     A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.5)
        # ], p=0.5),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    return transform


def xywh2xyxy(x, img_height, img_width):
    if isinstance(x, torch.Tensor):
        boxes = x.new(x.shape)
    elif isinstance(x, np.ndarray):
        boxes = np.zeros_like(x)
    else:
        raise TypeError("Input must be a PyTorch Tensor or a NumPy array")

    boxes[:, 0] = (x[:, 0] - x[:, 2] / 2) * img_width   # xmin
    boxes[:, 1] = (x[:, 1] - x[:, 3] / 2) * img_height  # ymin
    boxes[:, 2] = (x[:, 0] + x[:, 2] / 2) * img_width   # xmax
    boxes[:, 3] = (x[:, 1] + x[:, 3] / 2) * img_height  # ymax
    
    return boxes


def xyxy2xywh(boxes, img_height, img_width):
    dw, dh = 1. / img_width, 1. / img_height

    x_center = (boxes[:, 0] + boxes[:, 2]) / 2.0
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2.0
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    x_center *= dw
    y_center *= dh
    width *= dw
    height *= dh

    y = np.vstack((x_center, y_center, width, height)).T

    return y


def resize_image_and_boxes(image, boxes, new_size):
    height, width = image.shape[:2]
    boxes = xywh2xyxy(boxes, height, width)

    image = cv2.resize(image, (new_size, new_size))
    if boxes is not None:
        x_scale = new_size / width
        y_scale = new_size / height
        boxes[:, 0] = boxes[:, 0] * x_scale
        boxes[:, 1] = boxes[:, 1] * y_scale
        boxes[:, 2] = boxes[:, 2] * x_scale
        boxes[:, 3] = boxes[:, 3] * y_scale

    boxes = xyxy2xywh(boxes, new_size, new_size)

    return image, boxes

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
        targets = torch.zeros((0, 6))  # 빈 targets 텐서 초기화

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

                targets = torch.zeros((len(boxes), 6))
                targets[:, 1:] = torch.from_numpy(boxes)
        # 파일이 존재하지 않거나 빈 파일인 경우 targets는 빈 텐서로 유지.

        # 이미지를 텐서로 변환
        image_tensor = torch.tensor(image, dtype=torch.float32)

        return img_path, image_tensor, targets

    

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
                resized_images.append(torch.from_numpy(image) / 255)

                # 타겟에 배치 인덱스 추가
                target[:, 0] = i  # 첫 번째 열에 배치 인덱스 할당
                target[:, 2:] = torch.from_numpy(boxes)  # 바운딩 박스 좌표 업데이트
                resized_targets.append(target)
            else:
                # 이미지만 크기 조정 (타겟이 없는 경우)
                image = resize_image_and_boxes(image, None, new_size=self.img_size)[0]
                image = np.transpose(image, axes=(2, 0, 1))
                resized_images.append(torch.from_numpy(image) / 255)

        # 타겟 텐서 생성
        if len(resized_targets) > 0:
            targets = torch.cat(resized_targets, 0)
        else:
            targets = None

        self.batch_count += 1
        
        return paths, torch.stack(resized_images), targets