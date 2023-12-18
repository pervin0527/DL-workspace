import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(is_train, img_size):
    if is_train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2), p=0.5)
            ], p=1),

            # A.ElasticTransform(alpha=1.0, sigma=15.0, alpha_affine=50.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False, same_dxdy=False, p=0.4),

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

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=[]))

    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=[]))

    return transform