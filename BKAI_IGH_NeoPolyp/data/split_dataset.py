import random
from glob import glob

"""
Total data : 1000
Train | Valid Rate : 8 : 2
"""

def get_file_list(img_path):
    images = sorted(glob(f"{img_path}/*.jpeg"))

    return images


def write_txt(files, name):
    with open(f"./{name}.txt", "w") as f:
        for idx, file in enumerate(files):
            file_name = file.split("/")[-1].split(".")[0]
            f.write(file_name)

            if idx != len(files) - 1:
                f.write("\n")


if __name__ == "__main__":
    DATA_DIR = "/home/pervinco/Datasets/BKAI_IGH_NeoPolyp"
    IMG_DIR = f"{DATA_DIR}/train/train"
    
    train_rate = 0.8
    valid_rate = 1 - train_rate

    images = get_file_list(IMG_DIR)
    random.shuffle(images)

    split_idx = int(0.8 * len(images))
    train_files = images[:split_idx]
    valid_files = images[split_idx:]
    print(len(train_files), len(valid_files))

    write_txt(train_files, "train")
    write_txt(valid_files, "valid")