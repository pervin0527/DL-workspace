import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(is_train, img_size):
    if is_train:
        transform = A.Compose([
            A.OneOf([A.Resize(height=img_size, width=img_size, p=0.5),
                     A.RandomResizedCrop(height=img_size, width=img_size, p=0.5)], p=1),

            A.OneOf([A.ShiftScaleRotate(shift_limit_x=(-0.6, 0.06), shift_limit_y=(-0.06, 0.06),
                                        scale_limit=(-0.1, 0.1), rotate_limit=(-45, 45),
                                        border_mode=0,
                                        p=0.5),
                     A.HorizontalFlip(p=0.25),
                     A.VerticalFlip(p=0.25)], p=0.6),
            
            A.OneOf([A.Blur(p=0.2), A.GaussianBlur(p=0.2), A.GlassBlur(p=0.2), A.MotionBlur(p=0.2), A.GaussNoise(p=0.2)], p=0.3),
            
            A.OneOf([A.RandomSnow(p=0.5),
                     A.RandomShadow(p=0.5)], p=0.2),

            A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
                       ToTensorV2(p=1)], p=1)
        ])
    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1)
        ])

    return transform