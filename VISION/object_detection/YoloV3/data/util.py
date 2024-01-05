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