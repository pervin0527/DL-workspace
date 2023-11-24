import torch
import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator


def tokenize(df, ko_tokenizer, en_tokenizer):
    total_ko_tokens = []
    total_en_tokens = []
    for index, row in df.iterrows():
        ko, en = row["원문"], row["번역문"]
        
        ko_tokens = ["<sos>"] + ko_tokenizer.morphs(ko.strip()) + ["<eos>"]
        en_tokens = ["<sos>"] + en_tokenizer(en.strip()) + ["<eos>"]

        total_ko_tokens.append(ko_tokens)
        total_en_tokens.append(en_tokens)

    return total_ko_tokens, total_en_tokens


def build_vocab(data_tokens):
    vocab = build_vocab_from_iterator(data_tokens, specials=['<pad>', '<sos>', '<eos>', '<unk>'], min_freq=1)
    vocab.set_default_index(vocab['<unk>'])

    return vocab


def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]


class TranslationDataset(Dataset):
    def __init__(self, src_indices, trg_indices):
        self.src_indices = src_indices
        self.trg_indices = trg_indices

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, idx):
        src_sample = torch.tensor(self.src_indices[idx], dtype=torch.long)
        trg_sample = torch.tensor(self.trg_indices[idx], dtype=torch.long)
        
        return src_sample, trg_sample
    
    def collate_fn(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)  # 0은 패딩 인덱스
        trg_batch_padded = pad_sequence(trg_batch, batch_first=True, padding_value=0)

        src_batch_padded = src_batch_padded.view(-1, src_batch_padded.shape[0])
        trg_batch_padded = trg_batch_padded.view(-1, trg_batch_padded.shape[0])
        
        return src_batch_padded, trg_batch_padded