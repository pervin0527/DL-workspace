import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DOWNLOAD = False
DATA_DIR = "/home/pervinco/Datasets"
SAVE_DIR = "/home/pervinco/Models/Transformer"

EPOCHS = 1000
BATCH_SIZE = 128
NUM_WORKERS = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-9
ADAM_EPS = 5e-9
SCHEDULER_FACTOR = 0.9
SCHEDULER_PATIENCE = 10
WARM_UP_STEP = 100
DROP_PROB = 0.1

SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'de'
D_MODEL = 512
MAX_SEQ_LEN = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FEEDFORWARD_DIM = 2048