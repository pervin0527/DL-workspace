import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(is_train, img_size):
    if is_train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5)
            ], p=1),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),

            A.OneOf([
                A.Blur(p=0.5), 
                A.GaussianBlur(p=0.5), 
                A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=0.5)
            ], p=0.5),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
            ], p=0.5),

            A.OneOf([
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, p=0.5),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, p=0.5)
            ], p=0.5),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

    return transform
