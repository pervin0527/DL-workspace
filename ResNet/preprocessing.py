import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

def scale_jitter(image, min_size=256, max_size=480, crop_size=(224, 224)):
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

if __name__ == "__main__":
    
    data_dir = "/home/pervinco/Datasets/Stanford Car/train"
    images = sorted(glob(f"{data_dir}/*/*.jpg"))

    for image in images:
        print(image)
        image = cv2.imread(image)
        print(image.shape)
        result_image = scale_jitter(image)
        print(result_image.shape)

        cv2.imshow("result", result_image)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()