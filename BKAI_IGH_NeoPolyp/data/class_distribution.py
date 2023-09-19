import cv2
import numpy as np
from glob import glob

def check_mask_dist(file_path):
    mask = cv2.imread(file_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    total_red, total_green, total_black = 0, 0, 0
    red_mask = mask[:, :, 0] >= threshold
    green_mask = mask[:, :, 1] >= threshold
    black_mask = ~red_mask & ~green_mask

    red_count = np.sum(red_mask)
    green_count = np.sum(green_mask)
    black_count = np.sum(black_mask)    

    total_red += red_count
    total_green += green_count
    total_black += black_count

    return total_red, total_green, total_black


def write_file(file_name, file_list):
    with open(f"{dir}/{file_name}.txt", "w") as f:
        for idx, file in enumerate(file_list):
            print(file)
            f.write(file)

            if idx != len(file_list) - 1:
                f.write("\n")


if __name__ == "__main__":
    dir = "/home/pervinco/Datasets/BKAI_IGH_NeoPolyp"
    maks_dir = f"{dir}/train_gt"
    threshold = 100

    mask_files = sorted(glob(f"{maks_dir}/*.jpeg"))
    red_list, green_list, rg_list = [], [], []
    for file in mask_files:
        file_name = file.split('/')[-1].split('.')[0]
        red, green, _ = check_mask_dist(file)

        if red > 300 and green > 300:
            rg_list.append(file_name)
        
        elif red > 300:
            red_list.append(file_name)

        elif green > 300:
            green_list.append(file_name)

    print(len(red_list), len(green_list), len(rg_list))

    names = ["red", "green", "rng"]
    for i, file_list in enumerate([red_list, green_list, rg_list]):
        write_file(names[i], file_list)