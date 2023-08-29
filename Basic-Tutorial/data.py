import os
import cv2
import wget
import numpy as np


def download_dataset(urls, dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

    for url in urls:
        file_name = url.split('/')[-1]
        if not os.path.exists(f"{dir}/{file_name}"):
            wget.download(url, dir)

            print(f"{url} downloaded.")
        else:
            print(f"{url} already exists.")


def get_sample(csv_path):
    with open(csv_path, "r") as f:
        dataset = f.readlines()
    
    dataset = [data.strip() for data in dataset]
    print(len(dataset))

    data = dataset[0] ## 쉼표(,)가 포함되어 있기 때문에 길이가 1845.
    print(len(data))

    data = data.split(',')  ## 쉼표를 제거하고 나면 785가 된다.
    print(len(data))

    img_arr = np.asfarray(data[1:]).reshape((28,28)) 
    print(img_arr.shape)
    

if __name__ == "__main__":
    DATA_DIR = "/home/pervinco/Datasets/MNIST"
    URLS = ("https://pjreddie.com/media/files/mnist_train.csv", "https://pjreddie.com/media/files/mnist_test.csv")

    download_dataset(URLS, DATA_DIR)
    get_sample(f"{DATA_DIR}/mnist_train.csv")