import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(is_train, img_size):
    if is_train:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size, p=1),

            A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
                ToTensorV2(p=1)
            ], p=1)
        ], bbox_params=A.BboxParams(format="yolo", label_fields=['labels']))

    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size, p=1),

            A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), p=1),
                ToTensorV2(p=1)
            ], p=1)
        ], bbox_params=A.BboxParams(format="yolo", label_fields=['labels']))

    return transform