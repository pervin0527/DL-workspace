import cv2
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


def resize_image_and_boxes(image, boxes, new_size):
    # 이미지를 new_size로 리사이즈
    old_size = image.shape[:2]
    image = cv2.resize(image, new_size)

    # 바운딩 박스 좌표 조정
    if boxes is not None:
        x_scale = new_size[0] / old_size[1]
        y_scale = new_size[1] / old_size[0]
        boxes[:, 1] = boxes[:, 1] * x_scale
        boxes[:, 2] = boxes[:, 2] * y_scale
        boxes[:, 3] = boxes[:, 3] * x_scale
        boxes[:, 4] = boxes[:, 4] * y_scale

    return image, boxes