import torch
from torchvision.transforms import v2 as T

def get_transform(is_train, img_size):
    if is_train:
        transform = T.Compose([
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.486], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.Resize(img_size),
            T.ToDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.486], std=[0.229, 0.224, 0.225])
        ])

    return transform