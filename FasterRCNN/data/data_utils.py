import cv2
import torch
import random
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target
    
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)

        return image, target
    
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            height, width = image.shape[:2]
            image = image.numpy()
            image = torch.tensor(cv2.flip(image, 1))
            
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox

        return image, target
    
@torch.jit.script
class ImageList(object):
    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
