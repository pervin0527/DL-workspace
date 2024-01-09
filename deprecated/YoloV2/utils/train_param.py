import yaml

def read_train_params(yaml_file):
    with open(yaml_file, "r") as f:
        params = yaml.safe_load(f)
    
    return params

def save_train_params(save_dir, params):
    with open(f"{save_dir}/params.yaml", 'w') as file:
        yaml.dump(params, file)