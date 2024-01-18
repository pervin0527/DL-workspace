import torch
import logging

from datetime import timedelta
from torch import distributed as dist

from models.configs import vit_configs
from utils import set_seed, set_device, read_yaml_file

def define_vit(params):
    config = vit_configs(params["model_type"])
    num_classes = 10 if params["dataset"] == "cifar10" else 100
    


def main(params):
    if params["local_rank"] == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params.num_gpus = torch.cuda.device_count()
    else:
        torch.cuda.set_device(params["local_rank"])
        device = torch.device("cuda", params["local_rank"])
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60)) ## 프로세스 그룹을 초기화.
        params.num_gpus = 1 ## 분산 학습 환경에서는 각 프로세스가 하나의 GPU를 사용하므로, num_gpu를 1로 설정.
    
    params.device = device
    local_rank, num_gpus = params["local_rank"], params["num_gpus"]
    logger.warning(f"Process rank : {local_rank}, Device : {device}, Num_gpus : {num_gpus}, Distributed Training : {bool(local_rank != -1)}")

    set_seed(params["seed"])
    params, model = define_vit(params)


if __name__ == "__main__":
    params = read_yaml_file("./config.yaml")
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')