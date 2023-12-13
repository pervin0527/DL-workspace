import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(is_train, img_size):
    if is_train:
        transform = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2), p=0.5)
            ], p=1),

            A.OneOf([
                A.Resize(height=img_size, width=img_size, p=0.7),
                A.Compose([
                    A.PadIfNeeded(min_height=img_size * 2, min_width=img_size * 2, p=1),
                    A.RandomResizedCrop(height=img_size, width=img_size, p=1),
                ], p=0.3)
            ], p=1),

            A.OneOf([
                A.ShiftScaleRotate(shift_limit_x=(-0.6, 0.06), shift_limit_y=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-45, 45), border_mode=0, p=0.6),
                A.ElasticTransform(alpha=1.0, sigma=15.0, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False, same_dxdy=False, p=0.4),
            ], p=1),

            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)
            ], p=0.7),
            
            A.OneOf([
                A.Blur(p=0.2), 
                A.GaussianBlur(p=0.2), 
                A.GlassBlur(p=0.2), 
                A.MotionBlur(p=0.2), 
                A.GaussNoise(p=0.2)
            ], p=0.5),
            
            A.OneOf([
                A.RandomSnow(p=0.5),
                A.RandomShadow(p=0.5)
            ], p=0.5),

            A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
                ToTensorV2(p=1)
            ], p=1)
        ])
    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size, p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1)
        ])

    return transform