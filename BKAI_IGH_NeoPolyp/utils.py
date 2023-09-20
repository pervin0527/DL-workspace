import os
import cv2
import yaml
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def predict(epoch, config, img_size, model, device):
    model.eval()
    
    data_dir = config["data_dir"]
    save_dir = config["save_dir"]
    num_samples = config["num_pred_samples"]

    if not os.path.isdir(f"{save_dir}/predict"):
        os.makedirs(f"{save_dir}/predict")

    with open(f"{data_dir}/valid.txt", "r") as f:
        files = f.readlines()

    random.shuffle(files)
    samples = random.sample(files, num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 25))
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

        color_decoded = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        color_decoded[pred_mask == 0] = [0, 0, 0]
        color_decoded[pred_mask == 1] = [255, 0, 0]
        color_decoded[pred_mask == 2] = [0, 255, 0]
        color_decoded = cv2.resize(color_decoded, (mask.shape[1], mask.shape[0]))

        axes[idx, 0].imshow(overlayed)
        axes[idx, 0].set_title("Original Mask")

        axes[idx, 1].imshow(color_decoded)
        axes[idx, 1].set_title("Predict Mask")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/predict/epoch_{epoch:>04}.png")
    plt.close()


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


def save_config_to_yaml(config, save_dir):
    with open(f"{save_dir}/params.yaml", 'w') as file:
        yaml.dump(config, file)