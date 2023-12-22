import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, csv_path, grid_size=7, num_boxes=2, num_classes=20, transform=None):
        self.df = pd.read_csv(csv_path)
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx, 0]
        label_path = self.df.iloc[idx, 1]

        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in label.replace("\n", "").split()]
                boxes.append([class_label, x, y, width, height])
        
        image = Image.open(image_path)
        boxes = torch.tensor(boxes) ## [class_id, x, y, w, h]

        if self.transform:
            class_labels = boxes[:, 0].unsqueeze(1) ## [class_id]
            box_coords = boxes[:, 1:] ## [x, y, w, h]

            transformed = self.transform(image=np.array(image), bboxes=box_coords)
            image = transformed['image']

            transformed_boxes = torch.tensor(transformed['bboxes'])
            boxes = torch.cat((class_labels[:len(transformed_boxes)], transformed_boxes), dim=1)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist() ## class_id, center_x, center_y, box_width, box_height
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x) ## 박스의 중심 좌표를 그리드 셀 인덱스로 변환.
            x_cell, y_cell = self.S * x - j, self.S * y - i ## 박스의 중심 좌표를 셀 내 좌표로 변환.
            width_cell, height_cell = width * self.S, height * self.S ## 박스의 너비와 높이를 그리드 크기에 맞게 조정.

            ## grid cell
            ## 0 ~ 19 : classes

            ## 20 : confidence_score1
            ## 21 ~ 24 : box1 coord

            ## 25 : confidence_score2(zero) 
            ## 26 ~ 29 : box2 coord(zeros)
            
            ## 즉, 첫번째 box만이 object를 포함하는 중. YoloV1은 두 개 중 하나만이 object를 담을 수 있다.
            if label_matrix[i, j, 20] == 0: ## i,j번째 cell의 20번째 원소에 objectness를 표기.
                label_matrix[i, j, 20] = 1

                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates

                label_matrix[i, j, class_label] = 1

        return image, label_matrix



if __name__ == "__main__":
    dataset = VOCDataset("./train.csv")
    print(dataset[0].shape)