import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset

from data.util import pad_to_square


class YoloDataset(Dataset):
    def __init__(self, file_list_path, img_size, augment, multiscale, normalized_labels):
        self.img_size = img_size
        self.augment = augment
        self.max_objects = 100
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.batch_count = 0

        with open(file_list_path, 'r') as f:
            self.image_files = f.readlines()

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('JPEGImages', 'labels') for path in self.image_files]


    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        img_path = self.image_files[idx].rstrip()
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, c = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        image, pad = pad_to_square(image)
        _, padded_h, padded_w = image.shape

        targets = None
        label_path = self.label_files[idx].rstrip()
        if os.path.exists(label_path):
            boxes = np.loadtxt(label_path).reshape(-1, 5)

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

            targets = np.zeros((len(boxes), 6))
            targets[:, 1:] = boxes


        return img_path, image, targets
        

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        ## Remove empty placeholder targets
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
        # images = torch.stack([resize(image, self.image_size) for image in images])
        self.batch_count += 1

        return paths, images, targets