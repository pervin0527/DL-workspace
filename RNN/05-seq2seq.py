import random
import torch
from torch import nn
from torch import optim

def filter_pair(pair, source_max_length, target_max_length):
    return len(pair[0].split(" ") < source_max_length and len(pair[1].split(" "))) < target_max_length

def preprocess(corpus, source_max_length, target_max_length):
    print("reading courpus")
    
    pairs = []
    for line in corpus:
        pairs.append([s for s in line.strip().lower().split("\t")])
    print(f"Read {len(pairs)} sentence pairs.")

    pairs = [pair for pair in pairs if filter_pair]

class Vocab:
    def __init__(self):
        self.vocab2index = {"<SOS>" : SOS_token, "<EOS>" : EOS_token}
        self.index2vocab = {SOS_token : "<SOS>", EOS_token : "<EOS>"}
        self.vocab_count = {}
        self.n_vocab = len(self.vocab2index)

    def add_vocab(self, sentence):
        for word in sentence.split(" "):
            if word not in self.vocab2index:
                self.vocab2index[word] = self.n_vocab
                self.vocab_count[word] = 1
                self.index2vocab[self.n_vocab] = word
                self.n_vocab += 1
            else:
                self.vocab_count[word] += 1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw_data = ["I feel hungry. 나는 배가 고프다.",
                "Pytorch is very easy.  파이토치는 매우 쉽다.",
                "Pytorch is a framework for deep learning.  파이토치는 딥러닝을 위한 프레임워크이다.",
                "Pytorch is very clear to use. 파이토치는 사용하기 매우 직관적이다."]
    
    SOS_token = 0
    EOS_token = 1