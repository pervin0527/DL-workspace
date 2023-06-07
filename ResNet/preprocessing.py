import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

class ScaleJitter(object):
    def __init__(self, min_size, max_size, crop_size, p=0.5):
        self.min_size = min_size
        self.max_size = max_size
        self.crop_size = crop_size
        self.p = p

    def __call__(self, image):
        if self.p < np.random.random():
            return self.scale_jitter(image, self.min_size, self.max_size, self.crop_size)
        else:
            return image

    def scale_jitter(self, image, min_size=256, max_size=480, crop_size=(224, 224)):
        height, width = image.shape[:2]
        if height > width:
            width = np.random.randint(min_size, max_size+1)
        
        else:
            height = np.random.randint(min_size, max_size+1)
        image = cv2.resize(image, (width, height))

        try:
            y = np.random.randint(0, image.shape[0] - crop_size[0] + 1)
            x = np.random.randint(0, image.shape[1] - crop_size[1] + 1)
            image = image[y : y + crop_size[0], x : x + crop_size[1]]
        except:
            image = cv2.resize(image, crop_size)

        return image


def get_mean_std(train_ds_path):
    total_pixels = 0
    sum_pixels = np.zeros(shape=[3], dtype=np.float32)
    sum_diffs = np.zeros(shape=[3], dtype=np.float32)

    train_image_files = sorted(glob(f"{train_ds_path}/*/*.jpg"))
    ## mean per pixels
    for idx in tqdm(range(len(train_image_files))):
        image = cv2.imread(train_image_files[idx])
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0

        sum_pixels[0] += np.sum(image[:, :, 0]) ## Blue
        sum_pixels[1] += np.sum(image[:, :, 1]) ## Green
        sum_pixels[2] += np.sum(image[:, :, 2]) ## Red

        height, width, channels = image.shape
        total_pixels += (height * width * channels)

    mean_per_pixels = sum_pixels / total_pixels

    ## std per pixles
    for idx in tqdm(range(len(train_image_files))):
        image = cv2.imread(train_image_files[idx])
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0

        sum_diffs[0] += np.sum((image[:, :, 0] - mean_per_pixels[0]) ** 2)
        sum_diffs[1] += np.sum((image[:, :, 1] - mean_per_pixels[1]) ** 2)
        sum_diffs[2] += np.sum((image[:, :, 2] - mean_per_pixels[2]) ** 2)

    std_per_pixels = np.sqrt(sum_diffs / total_pixels)

    mean_per_pixels = np.full(3, np.sum(mean_per_pixels))
    std_per_pixels = np.full(3, np.sum(std_per_pixels))

    return mean_per_pixels, std_per_pixels


if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/sports/train"
    images = sorted(glob(f"{data_dir}/*/*.jpg"))

    mean_per_pixels, std_per_pixels = get_mean_std(data_dir)
    print(mean_per_pixels, std_per_pixels)