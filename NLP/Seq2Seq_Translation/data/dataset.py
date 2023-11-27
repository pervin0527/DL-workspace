import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

def build_vocab(texts, tokenizer):
    tokenized_texts = map(tokenizer, texts)
    vocab = build_vocab_from_iterator(tokenized_texts, specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=2)
    vocab.set_default_index(vocab['<unk>'])

    return vocab


def text_transform(sentence, tokenizer, vocab):
    tokens = [vocab[token] for token in tokenizer(sentence)]
    tokens.insert(0, vocab["<sos>"])
    tokens.append(vocab["<eos>"])
    
    return tokens

class TranslationDataset(Dataset):
    def __init__(self, dataset):
        self.src_indices, self.trg_indices = dataset[0], dataset[1]

    def __len__(self):
        return len(self.src_indices)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(self.src_indices[idx], dtype=torch.long)
        trg_tensor = torch.tensor(self.trg_indices[idx], dtype=torch.long)

        return src_tensor, trg_tensor
    

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, padding_value=0)
    trg_batch = pad_sequence(trg_batch, padding_value=0)

    return src_batch, trg_batch