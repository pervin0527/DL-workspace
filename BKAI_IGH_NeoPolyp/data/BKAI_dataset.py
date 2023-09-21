import cv2
import random
import numpy as np
import albumentations as A

from torch.utils.data import Dataset

class BKAIDataset(Dataset):
    def __init__(self, base_dir, split, size, threshold=50):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.size = size
        self.threshold = threshold

        self.set_txt = f"{base_dir}/{split}.txt"
        with open(self.set_txt, "r") as f:
            file_list = f.readlines()

        self.file_list = [x.strip() for x in file_list]

    
    def load_img_mask(self, index):
        file_name = self.file_list[index]
        image = cv2.imread(f"{self.base_dir}/train/{file_name}.jpeg")
        mask = cv2.imread(f"{self.base_dir}/train_gt/{file_name}.jpeg") 
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

        return image, mask
        

    def __getitem__(self, index):
        image, mask = self.load_img_mask(index)

        if self.split == "train":
            if random.random() < 0.5:
                transform_image, transform_mask = self.train_transform(image, mask)
            else:
                transform_image, transform_mask = self.mosaic_augmentation(image, mask)
        
        else:
            transform_image, transform_mask = self.valid_transform(image, mask)
        
        transform_image = np.transpose(transform_image, (2, 0, 1)) ## H, W, C -> C, H, W
        transform_image = transform_image / 255.0

        transform_mask = self.encode_mask(transform_mask)

        return transform_image, transform_mask
    

    def __len__(self):
        return len(self.file_list)


    def encode_mask(self, mask):
        label_transformed = np.full(mask.shape[:2], 0, dtype=np.uint8)

        red_mask = (mask[:, :, 0] > self.threshold) & (mask[:, :, 1] < 50) & (mask[:, :, 2] < 50)
        label_transformed[red_mask] = 1

        green_mask = (mask[:, :, 0] < 50) & (mask[:, :, 1] > self.threshold) & (mask[:, :, 2] < 50)
        label_transformed[green_mask] = 2

        return label_transformed
    

    def train_transform(self, image, mask):
        # transform = A.Compose([A.Flip(p=0.3),
        #                        A.ShiftScaleRotate(shift_limit_x=(-0.06, 0.06), shift_limit_y=(-0.06, 0.06), 
        #                                           scale_limit=(-0.3, 0.1), rotate_limit=(-90, 90),
        #                                           interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None,  rotate_method='largest_box', p=0.7)], p=1),
        # ])

        transform = A.Compose([A.HorizontalFlip(),
                               A.VerticalFlip(),
                               A.ColorJitter(brightness=(0.6,1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True),
                               A.Affine(scale=(0.5,1.5), translate_percent=(-0.125,0.125), rotate=(-180,180), shear=(-22.5,22), always_apply=True),])        
        
        transformed = transform(image=image, mask=mask)
        transformed_image, transformed_mask = transformed["image"], transformed["mask"]

        return transformed_image, transformed_mask
    

    def valid_transform(self, image, mask):
        transform = A.Compose([A.Resize(self.size, self.size, p=1)])
        
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        return transformed_image, transformed_mask
    

    def mosaic_augmentation(self, image, mask):
        h, w = self.size, self.size
        mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
        mosaic_mask = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)

        candidates = []
        is_full = False
        while not is_full:
            idx = random.randint(0, len(self.file_list)-1)

            if not idx in candidates:
                candidates.append(idx)

            if len(candidates) < 4:
                is_full = True
        
        indices = [0, 1, 2, 3]
        random.shuffle(indices)
        for i, index in enumerate(indices):
            image, mask = self.load_img_mask(index)
            
            if i == 0:
                mosaic_img[:cy, :cx] = cv2.resize(image, (cx, cy))
                mosaic_mask[:cy, :cx] = cv2.resize(mask, (cx, cy))
            elif i == 1:
                mosaic_img[:cy, cx:] = cv2.resize(image, (w-cx, cy))
                mosaic_mask[:cy, cx:] = cv2.resize(mask, (w-cx, cy))
            elif i == 2:
                mosaic_img[cy:, :cx] = cv2.resize(image, (cx, h-cy))
                mosaic_mask[cy:, :cx] = cv2.resize(mask, (cx, h-cy))
            elif i == 3:
                mosaic_img[cy:, cx:] = cv2.resize(image, (w-cx, h-cy))
                mosaic_mask[cy:, cx:] = cv2.resize(mask, (w-cx, h-cy))
        
        return mosaic_img, mosaic_mask
