import os
import torch

from datetime import datetime

from utils.util import read_yaml, save_yaml, make_log_dir
from models.darknet53 import build_darknet53


def main():
    now = datetime.now().strftime('%y%m%d_%H%M%S')
    print(now)

    train_path = cfg['train']
    valid_path = cfg['valid']
    class_names = cfg['names']
    print(f"{train_path}\n{valid_path}\n{class_names}")

    darknet53 = build_darknet53()
    dummy_input = torch.randn(1, 3, 256, 256)
    output = darknet53(dummy_input)

    print("Output shape:", output.shape)


if __name__ == "__main__":
    num_workers = os.cpu_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = read_yaml("./config.yaml")

    main()