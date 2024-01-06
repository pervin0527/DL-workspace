import cv2
import torch
import numpy as np

def pad_to_square(image, pad_value=0):
    h, w = image.shape[:2]

    # 너비와 높이의 차
    difference = abs(h - w)

    # (top, bottom) padding or (left, right) padding
    if h <= w:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [(0, 0), (top, bottom)]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [(left, right), (0, 0)]

    # Add padding
    image_padded = cv2.copyMakeBorder(image, pad[1][0], pad[1][1], pad[0][0], pad[0][1], cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])
    
    return image_padded


def xywh2xyxy(x, img_height, img_width):
    if isinstance(x, torch.Tensor):
        boxes = x.new(x.shape)
    elif isinstance(x, np.ndarray):
        boxes = np.zeros_like(x)
    else:
        raise TypeError("Input must be a PyTorch Tensor or a NumPy array")

    boxes[:, 0] = (x[:, 0] - x[:, 2] / 2) * img_width   # xmin
    boxes[:, 1] = (x[:, 1] - x[:, 3] / 2) * img_height  # ymin
    boxes[:, 2] = (x[:, 0] + x[:, 2] / 2) * img_width   # xmax
    boxes[:, 3] = (x[:, 1] + x[:, 3] / 2) * img_height  # ymax
    
    return boxes


def xyxy2xywh(boxes, img_height, img_width):
    dw, dh = 1. / img_width, 1. / img_height

    x_center = (boxes[:, 0] + boxes[:, 2]) / 2.0
    y_center = (boxes[:, 1] + boxes[:, 3]) / 2.0
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    x_center *= dw
    y_center *= dh
    width *= dw
    height *= dh

    y = np.vstack((x_center, y_center, width, height)).T

    return y


def resize_image_and_boxes(image, boxes, new_size):
    height, width = image.shape[:2]
    boxes = xywh2xyxy(boxes, height, width)

    image = cv2.resize(image, (new_size, new_size))
    if boxes is not None:
        x_scale = new_size / width
        y_scale = new_size / height
        boxes[:, 0] = boxes[:, 0] * x_scale
        boxes[:, 1] = boxes[:, 1] * y_scale
        boxes[:, 2] = boxes[:, 2] * x_scale
        boxes[:, 3] = boxes[:, 3] * y_scale

    boxes = xyxy2xywh(boxes, new_size, new_size)

    return image, boxes


def draw_boxes(image, boxes, class_idx, total_classes):
    height, width = image.shape[:2]
    for box, label in zip(boxes, class_idx):
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        label_str = f'{total_classes[int(label)]}'
        cv2.putText(image, label_str, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    # plt.axis('off')  # Hide axis
    # plt.show()
        
    cv2.imwrite("./sample.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))