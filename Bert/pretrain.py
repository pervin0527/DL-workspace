import torch
import argparse
import config as cfg
from data.vocab import load_vocab

def train_model(rank, world_size, args):
    print(rank, world_size, args)

    vocab = load_vocab(args.vocab)
    config = cfg.Config.load(args.config)
    config.n_enc_vocab = len(vocab)
    config.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None, type=int, required=False, help="GPU id to use.")
    parser.add_argument("--count", default=10, type=int, required=False, help="count of pretrain data")
    parser.add_argument("--epoch", default=20, type=int, required=False, help="epoch")
    parser.add_argument("--batch", default=256, type=int, required=False, help="batch")
    parser.add_argument('--learning_rate', type=float, default=5e-5, required=False, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0, required=False, help="weight decay")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, required=False, help="adam epsilon")
    parser.add_argument('--warmup_steps', type=float, default=0, required=False, help="warmup steps")
    parser.add_argument("--config", default="./config.json", type=str, required=False, help="config file")
    parser.add_argument("--vocab", default="./data/kowiki/kowiki.model", type=str, required=False, help="vocab file")
    parser.add_argument("--input", default="./data/kowiki/kowiki_bert_{}.json", type=str, required=False, help="input pretrain data file")
    parser.add_argument("--save", default="/home/pervinco/Models/BERT/save_pretrain.pth", type=str, required=False, help="save file")
    parser.add_argument('--seed', type=int, default=42, required=False, help="random seed for initialization")
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count() if args.gpu is None else 1
    else:
        args.n_gpu = 0

    # print(args)
    train_model(0 if args.gpu is None else args.gpu, args.n_gpu, args)