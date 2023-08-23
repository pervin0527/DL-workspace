import os
import torch
from torchtext import transforms
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from utils import data_download, load_pkl, save_pkl


class Multi30k:
    UNK, UNK_IDX = "<unk>", 0
    PAD, PAD_IDX = "<pad>", 1
    # SOS, SOS_IDX = "<sos>", 2
    # EOS, EOS_IDX = "<eos>", 3
    # SPECIALS = {UNK : UNK_IDX, PAD : PAD_IDX, SOS : SOS_IDX, EOS : EOS_IDX}
    SPECIALS = {UNK : UNK_IDX, PAD : PAD_IDX}

    def __init__(self, data_dir, target_language, max_seq_len, min_freq):
        self.data_dir = f"{data_dir}/Multi30k"
        self.cache_dir = f"{self.data_dir}/caches"

        if not os.path.isdir(self.data_dir):
            data_download(self.data_dir)

        self.target_language = target_language
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq

        self.target_tokenizer = self.build_tokenizer(self.target_language)

        self.train, self.valid, self.test = None, None, None
        self.build_dataset()

        self.vocab = None
        self.build_vocab()


    def build_tokenizer(self, language):
        spacy_lang_dict = {'en': "en_core_web_sm", 'de': "de_core_news_sm"}
        assert language in spacy_lang_dict.keys()

        return get_tokenizer("spacy", spacy_lang_dict[language])


    def build_dataset(self):
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        train_pkl = f"{self.cache_dir}/train.pkl"
        if os.path.exists(train_pkl):
            self.train = load_pkl(train_pkl)

        else:
            with open(f"{self.data_dir}/train.en") as f:
                self.train = [text.rstrip() for text in f]
        
            save_pkl(self.train, train_pkl)

        val_pkl = f"{self.cache_dir}/val.pkl"
        if os.path.exists(val_pkl):
            self.valid = load_pkl(val_pkl)
            
        else:
            with open(f"{self.data_dir}/val.en") as f:
                self.valid = [text.rstrip() for text in f]
        
            save_pkl(self.valid, val_pkl)

        test_pkl = f"{self.cache_dir}/test.pkl"
        if os.path.exists(test_pkl):
            self.test = load_pkl(test_pkl)
            
        else:
            with open(f"{self.data_dir}/test.en") as f:
                self.test = [text.rstrip() for text in f]
        
            save_pkl(self.test, test_pkl)


    def build_vocab(self):
        assert self.train is not None

        char_set = set()
        for sentence in self.train:
            char_set.update(sentence)
        char_set = sorted(list(char_set))
        vocab = {c: i for i, c in enumerate(char_set)}

        self.vocab = vocab


    def get_dataset(self):
        return self.train, self.valid, self.test, self.vocab


def preprocessing(sentence, str_vocabs):
    x = torch.zeros(1, len(sentence) - 1, len(str_vocabs))
    y = torch.zeros(1, len(sentence) - 1, dtype=torch.int64)

    x_str = sentence[:-1]
    y_str = sentence[1:]

    for idx, char in enumerate(x_str):
        x[0][idx][str_vocabs.index(char)] = 1

    for idx, char in enumerate(y_str):
        y[0][idx] = str_vocabs.index(char)

    return x, y