import os
import torch
import pickle

from torch import nn
from torchtext import transforms
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from utils import data_download, load_pkl, save_pkl

class Multi30k:
    UNK, UNK_IDX = "<unk>", 0
    PAD, PAD_IDX = "<pad>", 1
    SOS, SOS_IDX = "<sos>", 2
    EOS, EOS_IDX = "<eos>", 3
    SPECIALS = {UNK : UNK_IDX, PAD : PAD_IDX, SOS : SOS_IDX, EOS : EOS_IDX}

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

        self.vocab_transform = None
        self.build_transform()


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
            self.train = [x for x in self.train]
        
            save_pkl(self.train, train_pkl)

        val_pkl = f"{self.cache_dir}/val.pkl"
        if os.path.exists(val_pkl):
            self.val = load_pkl(val_pkl)
            
        else:
            with open(f"{self.data_dir}/val.en") as f:
                self.val = [text.rstrip() for text in f]
            self.val = [x for x in self.val]
        
            save_pkl(self.val, val_pkl)

        test_pkl = f"{self.cache_dir}/test.pkl"
        if os.path.exists(test_pkl):
            self.test = load_pkl(test_pkl)
            
        else:
            with open(f"{self.data_dir}/test.en") as f:
                self.test = [text.rstrip() for text in f]
            self.test = [x for x in self.test]
        
            save_pkl(self.test, test_pkl)


    def build_vocab(self):
        assert self.train is not None

        def yield_tokens():
            yield [str(token) for token in self.target_tokenizer(self.train)]

        print(self.train)
        vocab_file = f"{self.cache_dir}/vocab_{self.target_language}.pkl"
        if os.path.exists(vocab_file):
            vocab = load_pkl(vocab_file)
        else:
            vocab = build_vocab_from_iterator(yield_tokens(), min_freq=self.min_freq, specials=self.SPECIALS.keys())
            vocab.set_default_index(self.UNK_IDX)
            save_pkl(vocab, vocab_file)

        self.vocab = vocab


    def build_transform(self):
        def get_transform(self, vocab):
            return transforms.Sequential(transforms.VocabTransform(vocab),
                                         transforms.Truncate(self.max_seq_len-2),
                                         transforms.AddToken(token=self.SOS_IDX, begin=True),
                                         transforms.AddToken(token=self.EOS_IDX, begin=False),
                                         transforms.ToTensor(padding_value=self.PAD_IDX))

        self.vocab_transform = get_transform(self, self.vocab)


    def collate_fn(self, batch):
        trg = [self.target_tokenizer(data) for data in batch]
        batch_trg = self.vocab_transform(trg)

        return batch_trg, batch


    def get_iter(self, **kwargs):
        if self.vocab_transform is None:
            self.build_transform()

        train_iter = DataLoader(self.train, collate_fn=self.collate_fn, **kwargs)
        valid_iter = DataLoader(self.valid, collate_fn=self.collate_fn, **kwargs)
        test_iter = DataLoader(self.test, collate_fn=self.collate_fn, **kwargs)

        return train_iter, valid_iter, test_iter
    
if __name__ == "__main__":
    DATASET = Multi30k("/home/pervinco/Datasets/test", "en", 256, 2)