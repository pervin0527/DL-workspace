import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm


def make_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_file_list(dir):
    file_list = sorted(glob(f"{dir}/*.jpeg"))

    return file_list


def make_label_mask(file_list, dir):
    make_dir(dir)

    for idx in tqdm(range(len(file_list))):
        file = file_list[idx]
        file_name = file.split('/')[-1].split('.')[0]
        mask = cv2.imread(file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        label_transformed = np.full(mask.shape[:2], 2, dtype=np.uint8)  # 초기 마스크를 2(background)로 설정

        # Red color (neoplastic polyps)를 0으로 변환
        red_mask = (mask[:, :, 0] > 100) & (mask[:, :, 1] < 100) & (mask[:, :, 2] < 100)
        label_transformed[red_mask] = 0

        # Green color (non-neoplastic polyps)를 1로 변환
        green_mask = (mask[:, :, 0] < 100) & (mask[:, :, 1] > 100) & (mask[:, :, 2] < 100)
        label_transformed[green_mask] = 1

        cv2.imwrite(f"{dir}/{file_name}.jpeg", label_transformed)


def validate(label_dir, mask_dir):
    make_dir(PLOT_DIR)
    label_files = get_file_list(label_dir)
    mask_files = get_file_list(mask_dir)

    for idx in tqdm(range(len(label_files))):
        label_file, mask_file = label_files[idx], mask_files[idx]

        # label = cv2.imread(label_file)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        color_decoded = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        color_decoded[label == 0] = [255, 0, 0]  # Neoplastic (Red)
        color_decoded[label == 1] = [0, 255, 0]  # Non-neoplastic (Green)
        color_decoded[label == 2] = [0, 0, 0]  # Background (Black)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(mask)
        plt.title('Original Mask')

        plt.subplot(1, 2, 2)
        plt.imshow(color_decoded)
        plt.title('Decoded Label')
        
        save_path = os.path.join(PLOT_DIR, f"comparison_{idx}.png")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    DATA_DIR = "/home/pervinco/Datasets/BKAI_IGH_NeoPolyp"
    MASK_DIR = f"{DATA_DIR}/train_gt/train_gt"
    OUTPUT_DIR = f"{DATA_DIR}/train_label/train_label"
    PLOT_DIR = f"{DATA_DIR}/train_label/plots"

    mask_files = get_file_list(MASK_DIR)
    make_label_mask(mask_files, OUTPUT_DIR)
    validate(OUTPUT_DIR, MASK_DIR)