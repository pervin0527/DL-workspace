import os
import yaml
import torch
import random
import logging
import numpy as np

logger = logging.getLogger(__name__)

def set_seed(params):
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    if params["num_gpus"] > 0:
        torch.cuda.manual_seed_all(params["seed"])
 

def read_yaml_file(file_path):
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    
    return params

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_model(params, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(params["save_dir"], "train_checkpoint.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", params["save_dir"])