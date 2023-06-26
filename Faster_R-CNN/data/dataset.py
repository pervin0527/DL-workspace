import cv2
import torch
import random
from utils.config import opt
from skimage import transform as sktrn
from torchvision import transforms as tvtrn
from voc_dataset import PascalVoc

def preprocessing(image, min_size, max_size):
    C, H, W =  image.shape

    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)

    image = image / 255.
    image = sktrn.resize(image, (C, H * scale, W * scale), mode="reflect", anti_aliasing=False)
    normalize = tvtrn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(torch.from_numpy(image))

    return image

class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, data):
        image, bbox, label = data
        _, H, W = image.shape
        
        image = preprocessing(image, self.min_size, self.max_size)
        o_C, o_H, o_W = image.shape
        scale = o_H / H
        bbox = self.resize_bbox(bbox, (H, W), (o_H, o_W))

        image, params = self.random_flip(image, x_random=True, return_param=True)
        bbox = self.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return image, bbox, label, scale
    
    def resize_bbox(self, in_size, out_size):
        bbox = bbox.copy()
        x_scale = float(out_size[1]) / in_size[1]
        y_scale = float(out_size[0]) / in_size[0]

        bbox[:, 0] = y_scale * bbox[:, 0]
        bbox[:, 2] = y_scale * bbox[:, 2]
        bbox[:, 1] = x_scale * bbox[:, 1]
        bbox[:, 3] = x_scale * bbox[:, 3]

        return bbox
    
    def flip_bbox(self, bbox, size, y_flip=False, x_flip=False):
        H, W = size
        bbox = bbox.copy()
        if y_flip:
            y_max = H - bbox[:, 0]
            y_min = H - bbox[:, 2]
            bbox[:, 0] = y_min
            bbox[:, 2] = y_max
        if x_flip:
            x_max = W - bbox[:, 1]
            x_min = W - bbox[:, 3]
            bbox[:, 1] = x_min
            bbox[:, 3] = x_max
        
        return bbox
    
    def random_flip(self, image, y_random=False, x_random=False, return_param=False, copy=False):
        y_flip, x_flip = False, False
        if y_random:
            y_flip = random.choice([True, False])
        if x_random:
            x_flip = random.choice([True, False])
        
        if y_flip:
            img = img[:, ::-1, :]
        if x_flip:
            img = img[:, :, ::-1]

        if copy:
            img = img.copy()

        if return_param:
            return img, {'y_flip':y_flip, 'x_flip':x_flip}
        else:
            return img

class TrainDataset:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = PascalVoc(opt.DATA_DIR)
        self.transform = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        image, bbox, label, difficult = self.dataset[idx]
        image, bbox, label, scale = self.transform((image, bbox, label))

        return image.copy(), bbox.copy(), label.copy(), scale
    
    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    test = TrainDataset()
    print(test)