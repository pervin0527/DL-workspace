import torch
import random
import xml.etree.ElementTree as ET

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional as F


def resize(image, size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)


def generate_letter_box_image(image, fill_value=0):
    channel, height, width = image.shape
    difference = abs(height - width)

    if height <= width:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [0, 0, top, bottom]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [left, right, 0, 0]

    image = F.pad(image, pad, mode="constant", value=fill_value)

    return image, pad


class DetectionDataset(Dataset):
    def __init__(self, data_dir, set_name, annot_type=None, year=None, image_size=416, transform=None, multiscale=True, normalized_labels=True):
        self.data_dir = data_dir
        self.annot_type = annot_type
        self.set_name = set_name
        self.image_size = image_size
        self.transform = transform
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.batch_count=0

        with open("./classes.txt", "r") as f:
            self.classes = sorted([x.strip() for x in f.readlines()])
        
        if annot_type.lower() == "voc":
            if self.set_name == "train":
                self.set_name = "trainval"

            elif self.set_name == "valid":
                self.set_name = "val"    

            self.txt_path = f"{data_dir}/ImageSets/Main/{self.set_name}.txt"
            with open(self.txt_path, "r") as f:
                self.file_list = [x.strip() for x in f.readlines()]

        elif annot_type.lower() == "coco":
            if year != None:
                self.json_path = f"{data_dir}/annotations/instances_{set_name}{year}.json"

        self.dataset = self.get_dataset()


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        data = self.dataset[index]
        file_name = data["file_name"]
        image = data["image"]
        targets = data["targets"]

        image = self.transform(image)
        _, h, w = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        image, pad = generate_letter_box_image(image)
        _, padded_h, padded_w = image.shape

        bboxes = torch.tensor(targets).reshape(-1, 5)
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (bboxes[:, 1] - bboxes[:, 3] / 2)
        y1 = h_factor * (bboxes[:, 2] - bboxes[:, 4] / 2)
        x2 = w_factor * (bboxes[:, 1] + bboxes[:, 3] / 2)
        y2 = h_factor * (bboxes[:, 2] + bboxes[:, 4] / 2)

        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]

        # Returns (x, y, w, h)
        bboxes[:, 1] = ((x1 + x2) / 2) / padded_w
        bboxes[:, 2] = ((y1 + y2) / 2) / padded_h
        bboxes[:, 3] *= w_factor / padded_w
        bboxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(bboxes), 6))
        targets[:, 1:] = bboxes

        return file_name, image, targets
    

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Selects new image size every 10 batches
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(320, 608 + 1, 32))

        # Resize images to input shape
        images = torch.stack([resize(image, self.image_size) for image in images])
        self.batch_count += 1

        return paths, images, targets


    def get_dataset(self):
        dataset = []
        if self.annot_type == "voc":
            # for idx in tqdm(range(len(self.file_list))):
                # file = self.file_list[idx]
            for file in self.file_list:
                xml_path = f"{self.data_dir}/Annotations/{file}.xml"
                bboxes, labels, isdifficult = self.read_xml(xml_path)

                image_path = f"{self.data_dir}/JPEGImages/{file}.jpg"
                image = Image.open(image_path).convert("RGB")
                
                targets = []
                for bbox, label in zip(bboxes, labels):
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    label = self.classes.index(label)
                    targets.append([label, x, y, w, h])
                
                dataset.append({"file_name" : file, "image" : image, "targets" : targets})
                
        return dataset


    def voc_coord_transform(self, width, height, xmin, xmax, ymin, ymax):
        dw = 1. / width
        dh = 1. / height
        x = (xmin + xmax) / 2.0 - 1
        y = (ymin + ymax) / 2.0 - 1
        
        w = xmax - xmin
        h = ymax - ymin
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh

        return x, y, w, h


    def read_xml(self, path):
        xml_data = ET.parse(path)
        root = xml_data.getroot()

        size = root.find("size")
        width, height = int(size.find("width").text), int(size.find("height").text)
        
        bboxes, labels, isdifficult = [], [], []
        for obj in root.findall("object"):
            isdifficult.append(int(obj.find("difficult").text))
            
            name = obj.find("name").text.lower().strip()
            labels.append(name)

            bndbox = obj.find("bndbox")
            xmin = max((0, float(bndbox.find("xmin").text)))
            ymin = max((0, float(bndbox.find("ymin").text)))
            xmax = min((width - 1, xmin + max(0, float(bndbox.find("xmax").text) - 1)))
            ymax = min((height - 1, ymin + max(0, float(bndbox.find("ymax").text) - 1)))
            area = (xmax - xmin) * (ymax - ymin)
            
            if area > 0 and xmax > xmin and ymax > ymin:
                x, y, w, h = self.voc_coord_transform(width, height, xmin, xmax, ymin, ymax)
                bboxes.append([x, y, w, h])

        return bboxes, labels, isdifficult