import os
import csv

## 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
## 2007_test.txt > test.txt

def make_csv(csv_path, txt_file):
    if isinstance(txt_file, str):
        data = open(f"{data_dir}/{test_data_file}").readlines()
    elif isinstance(txt_file, list):
        data = []
        for path in txt_file:
            data.extend(open(f"{data_dir}/{path}").readlines())

    with open(csv_path, mode="w", newline="") as file:
        for line in data:
            image_file = line.split("/")[-1].replace("\n", "")
            text_file = image_file.replace(".jpg", ".txt")
            tmp = [image_file, text_file]
            writer = csv.writer(file)
            writer.writerow(tmp)


# read_train = open("train.txt", "r").readlines()

# with open("train.csv", mode="w", newline="") as train_file:
#     for line in read_train:
#         image_file = line.split("/")[-1].replace("\n", "")
#         text_file = image_file.replace(".jpg", ".txt")
#         data = [image_file, text_file]
#         writer = csv.writer(train_file)
#         writer.writerow(data)

# read_train = open("test.txt", "r").readlines()

# with open("test.csv", mode="w", newline="") as train_file:
#     for line in read_train:
#         image_file = line.split("/")[-1].replace("\n", "")
#         text_file = image_file.replace(".jpg", ".txt")
#         data = [image_file, text_file]
#         writer = csv.writer(train_file)
#         writer.writerow(data)

if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit"
    train_data_files = ["VOC2007/2007_train.txt", "VOC2007/2007_val.txt", "VOC2012/2012_train.txt", "VOC2012/2012_val.txt"]
    test_data_file = "VOC2007/2007_test.txt"

    make_csv("./train.csv", train_data_files)
    make_csv("./test.csv", test_data_file)