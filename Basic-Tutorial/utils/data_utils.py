import os
import wget
import pickle
import tarfile
import numpy as np

def download_and_extract(path):
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_name = os.path.join(path, url.split('/')[-1])
    wget.download(url, file_name)

    if file_name.endswith(".tar.gz"):
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=path)

        print("\n파일 다운로드 및 압축 해제 완료.")
    else:
        print("\n압축 파일이 아닙니다.")


def unpickle(files):
    dataset = {'data': [], 'labels': []}
    for file in files:
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            images = data_dict[b'data']
            labels = data_dict[b'labels']
            dataset['data'].extend(images)
            dataset['labels'].extend(labels)
    dataset['data'] = np.array(dataset['data'])
    dataset['labels'] = np.array(dataset['labels'])
    
    return dataset


def vec2img(images):
    X = []
    for image in images:
        ## 처음 1024개의 요소가 빨간색 채널, 다음 1024개의 요소가 녹색 채널, 마지막 1024개의 요소가 파란색 채널.
        image = np.reshape(image, (3, 32, 32)) # 이미지를 (3, 32, 32)로 재구성
        image = np.transpose(image, (1, 2, 0)) # # 축을 재배열하여 (32, 32, 3)으로 변환
        image = image.astype(np.uint8)
        X.append(image)

    return np.array(X)

def cat_filter(labels, target=3):
    Y = []
    for label in labels:
        if label == target:
            Y.append([1])
        else:
            Y.append([0])

    return np.array(Y).transpose((1, 0))


if __name__ == "__main__":
    download_and_extract("/home/pervinco/Datasets/test")