import torch
import random
from utils.config import opt
from data.voc_dataset import PascalVoc
from data.utils import random_flip, flip_bbox, resize_bbox
from skimage import transform as sktsf
from torchvision import transforms as tvtsf

def pytorch_normalze(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)

    normalize = pytorch_normalze
    return normalize(img)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)

        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = random_flip(img, x_random=True, return_param=True)
        bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale
    
class TrainDataset:
    def __init__(self, opt):
        self.opt = opt
        self.dataset = PascalVoc(opt.voc_data_dir)
        self.transform = Transform(opt.min_size, opt.max_size)


    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.dataset[idx]
        img, bbox, label, scale = self.transform((ori_img, bbox, label))

        return img.copy(), bbox.copy(), label.copy(), scale
    
    def __len__(self):
        return len(self.dataset)