import torch
import logging

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


def get_dataloader(params):
    if params["local_rank"] not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((params["img_size"], params["img_size"]), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((params["img_size"], params["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if params["dataset"] == "cifar10":
        trainset = datasets.CIFAR10(root=params["dataset_dir"],
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=params["dataset_dir"],
                                   train=False,
                                   download=True,
                                   transform=transform_test) if params["local_rank"] in [-1, 0] else None

    else:
        trainset = datasets.CIFAR100(root=params["dataset_dir"],
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root=params["dataset_dir"],
                                    train=False,
                                    download=True,
                                    transform=transform_test) if params["local_rank"] in [-1, 0] else None
    if params["local_rank"] == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if params["local_rank"] == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_dataloader = DataLoader(trainset,
                                  sampler=train_sampler,
                                  batch_size=params["train_batch_size"],
                                  num_workers=4,
                                  pin_memory=True)
    
    valid_dataloader = DataLoader(testset,
                                  sampler=test_sampler,
                                  batch_size=params["valid_batch_size"],
                                  num_workers=4,
                                  pin_memory=True) if testset is not None else None

    return train_dataloader, valid_dataloader
