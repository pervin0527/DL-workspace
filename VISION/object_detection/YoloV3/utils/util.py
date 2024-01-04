import os
import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        contents = yaml.safe_load(f)
    
    return contents


def save_yaml(save_dir, contents):
    with open(f"{save_dir}/config.yaml", "w") as f:
        yaml.dump(contents, f)


def make_log_dir(save_dir, record_contents=None):
    if not os.path.isdir(save_dir):
        os.makedirs(f"{save_dir}/weights")
        os.makedirs(f"{save_dir}/logs")
        
        if record_contents is not None:
            save_yaml(save_dir, record_contents)