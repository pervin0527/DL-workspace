import os
import torch

MAX_ITER = 10000
LEARNING_RATE = 0.001
NUM_LAYERS = 1

MAX_SEQ_LEN = 256
MIN_FREQ = 1
TARGET_LANGUAGE = 'en'
DATA_DIR = "/home/pervinco/Datasets/Vocab_Prediction"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")