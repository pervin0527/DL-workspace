import os
import torch

EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 256
MIN_FREQ = 1
TARGET_LANGUAGE = 'en'
DATA_DIR = "/home/pervinco/Datasets/Vocab_Prediction"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])