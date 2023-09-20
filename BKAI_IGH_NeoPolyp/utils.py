import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def predict(epoch, data_dir, img_size, model, device):
    if not os.path.isdir(f"./predict"):
        os.makedirs(f"./predict")

    with open(f"{data_dir}/valid.txt", "r") as f:
        files = f.readlines()

    random.shuffle(files)
    samples = random.sample(files, 5)
    for idx, sample in enumerate(samples):
        file = sample.strip()

        image = cv2.imread(f"{data_dir}/train/{file}.jpeg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(f"{data_dir}/train_gt/{file}.jpeg")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        overlayed = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

        x = cv2.resize(image, (img_size, img_size))
        x = np.transpose(x, (2, 0, 1))  ## H, W, C -> C, H, W
        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x) / 255.0
        x = x.to(device)

        y_pred = model(x)

        pred_mask = torch.argmax(y_pred[0], dim=0).cpu().numpy()

        # y_pred = torch.cat(y_pred, dim=1)
        # _, pred_mask = torch.max(y_pred, dim=1)
        # pred_mask = pred_mask.squeeze(0).cpu().numpy()

        color_decoded = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        color_decoded[pred_mask == 0] = [0, 0, 0]
        color_decoded[pred_mask == 1] = [255, 0, 0]
        color_decoded[pred_mask == 2] = [0, 255, 0]
        color_decoded = cv2.resize(color_decoded, (mask.shape[1], mask.shape[0]))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(overlayed)
        plt.title("Original Mask")

        plt.subplot(1, 2, 2)
        plt.imshow(color_decoded)
        plt.title("Predict Mask")

        plt.savefig(f"./predict/pred_{idx}.png")
        plt.close()


def calculate_weigths_labels(save_dir, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    
    # Initialize tqdm
    print("\nCalculating classes weights")

    for sample in dataloader:
        y = sample["mask"]
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l

    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)

    print(f"Done. {ret}")

    classes_weights_path = f"{save_dir}/class_weights.npy"
    np.save(classes_weights_path, ret)

    return ret

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")