import PIL
from torchvision import transforms

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_transform(set_name, img_size):
    if set_name == "train":
        TRANSFORM = transforms.Compose([transforms.RandomResizedCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        NORMALIZE])
    else:
        TRANSFORM = transforms.Compose([transforms.Resize(img_size, interpolation=PIL.Image.BICUBIC),
                                        transforms.ToTensor(),
                                        NORMALIZE])
        
    return TRANSFORM