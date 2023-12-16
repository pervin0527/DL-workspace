import cv2
import torch

from torch.utils.data import Dataset

class VOCDetectionDataset(Dataset):
    def __init__(self, file_path, grid_scale=7, num_boxes=2, num_classes=20, transform=None):
        self.files = open(file_path, "r").read().strip().split("\n")
        self.grid_scale = grid_scale
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path = self.files[idx]
        
        file_name = image_path.split("/")[-1].split('.')[0]
        folder_path = '/'.join(image_path.split("/")[:-2])
        label_path = f"{folder_path}/yolov1/{file_name}.txt"

        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()]
                boxes.append([class_label, x, y, width, height])

        image = cv2.imread(image_path)
        boxes = torch.tensor(boxes)

        if self.transform is not None:
            labels = boxes[:, 0].tolist()
            boxes = boxes[:, 1:].tolist()
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            transformed_boxes = transformed['bboxes']
            transformed_labels = transformed['labels']

            boxes = []
            for box, label in zip(transformed_boxes, transformed_labels):
                new_box = [label] + list(box)
                boxes.append(new_box)

            boxes = torch.tensor(boxes, dtype=torch.float32)

        ## 0 ~ 19 : 클래스 수
        ## 20 : 클래스 포함 여부
        ## 21 ~ 24 : x, y, w, h
        ## 25 : 클래스 포함 여부 ---> 0
        ## 26 ~ 29 : x, y, w, h ---> 0, 0, 0, 0
        label_matrix = torch.zeros((self.grid_scale, self.grid_scale, self.num_boxes * 5 + self.num_classes))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.grid_scale * y), int(self.grid_scale * x)
            x_cell, y_cell = self.grid_scale * x - j, self.grid_scale * y - i

            width_cell, height_cell = (width * self.grid_scale, height * self.grid_scale)

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1

                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates

                label_matrix[i, j, class_label] = 1

        return image, label_matrix


if __name__ == "__main__":
    dataset = VOCDetectionDataset("./train.txt")
    image, label_matrix = dataset[0]
    print(image.shape, label_matrix.shape)

    for i in range(7):
        for j in range(7):
            line = label_matrix[i][j]
            print(line)