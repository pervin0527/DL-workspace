import os
import math
import pandas as pd

def write_file_list(path, name, data):
    img_dir = f"{path}/train_images"

    dataset = []
    with open(f"{path}/{name}.txt", "w") as file:
        for item in data:
            img_file, labels = item
            line = f"{img_dir}/{img_file},{labels}"
            file.write(f"{line}\n")

            dataset.append(line)
    
    return dataset


def read_file_list(path):
    with open(path, "r") as file:
        data = [x.strip() for x in file.readlines()]

    return data


def get_datasets(data_dir, valid_ratio):
    if not os.path.exists(f"{data_dir}/train.txt") and not os.path.exists(f"{data_dir}/valid.txt"):
        print("make train.txt, valid.txt.")
        train_df = pd.read_csv(f"{data_dir}/train.csv") ## heads : ["image", "labels"]
    
        images = train_df["image"].to_list()
        labels = train_df["labels"].to_list()

        total_dataset = list(zip(images, labels))
        num_valid = math.ceil(len(total_dataset) * valid_ratio)
        num_train = len(total_dataset) - num_valid

        train_dataset = total_dataset[:num_train]
        valid_dataset = total_dataset[num_train:]

        train_dataset = write_file_list(data_dir, "train", train_dataset)
        valid_dataset = write_file_list(data_dir, "valid", valid_dataset)

    else:
        print("train.txt, valid.txt exsist.")
        train_dataset = read_file_list(f"{data_dir}/train.txt")
        valid_dataset = read_file_list(f"{data_dir}/valid.txt")

    return train_dataset, valid_dataset