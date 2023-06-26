from utils.config import opt
from data.dataset import TrainDataset

def train(**kwargs):
    opt.parse(kwargs)

    train_dataset = TrainDataset(opt)
    print(train_dataset[0])

if __name__ == "__main__":
    train()