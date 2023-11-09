import os

def make_save_dir(path):
    if not os.path.isdir(path):
        os.makedirs(f"{path}/weights")
        os.makedirs(f"{path}/logs")