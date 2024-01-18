import yaml
import torch
import random
import numpy as np


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