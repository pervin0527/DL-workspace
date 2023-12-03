import torch
import spacy
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k, multi30k
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset


multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"


class Multi30kDataset:
    UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ["<unk>", "<pad>", "<sos>", "<eos>"]

    def __init__(self, ext, tokenize_en, tokenize_de, sos_token, eos_token, batch_size, batch_first, device):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.device = device
        print('Dataset initializing start')

        self.token_transform = {ext[0]: tokenize_de, ext[1]: tokenize_en}
        self.train_data, self.valid_data = self.make_dataset()

        self.build_vocab()
        print('Dataset initializing done')


    def make_dataset(self):
        if self.ext == ('de', 'en'):
            self.tokenizer_src = self.tokenize_de
            self.tokenizer_trg = self.tokenize_en
        elif self.ext == ('en', 'de'):
            self.tokenizer_src = self.tokenize_en
            self.tokenizer_trg = self.tokenize_de
        else:
            raise ValueError("Invalid extension for source and target language")

        train_data, valid_data = Multi30k(split=('train', 'valid'), language_pair=(self.ext[0], self.ext[1]))
        train_data = to_map_style_dataset(train_data)
        valid_data = to_map_style_dataset(valid_data)

        return train_data, valid_data
    

    def build_vocab(self):
        self.src_vocab = build_vocab_from_iterator(map(lambda x: self.tokenizer_src(x[0]), self.train_data), specials=self.special_symbols)
        self.trg_vocab = build_vocab_from_iterator(map(lambda x: self.tokenizer_trg(x[1]), self.train_data), specials=self.special_symbols)
        self.src_vocab.set_default_index(self.src_vocab[self.sos_token])
        self.trg_vocab.set_default_index(self.trg_vocab[self.sos_token])


    def collate_fn(self, batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_tokenized = self.token_transform[self.ext[0]](src_sample.rstrip("\n"))
            trg_tokenized = self.token_transform[self.ext[1]](trg_sample.rstrip("\n"))

            src_batch.append(torch.cat((torch.tensor([self.src_vocab[self.sos_token]]), torch.tensor([self.src_vocab[token] for token in src_tokenized]), torch.tensor([self.src_vocab[self.eos_token]]))))
            trg_batch.append(torch.cat((torch.tensor([self.trg_vocab[self.sos_token]]), torch.tensor([self.trg_vocab[token] for token in trg_tokenized]), torch.tensor([self.trg_vocab[self.eos_token]]))))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        trg_batch = pad_sequence(trg_batch, padding_value=self.PAD_IDX)


        if self.batch_first:
            batch_size = src_batch.size(1)
            src_batch = src_batch.view(batch_size, -1)
            trg_batch = trg_batch.view(batch_size, -1)

        return src_batch, trg_batch


    def make_iter(self):
        train_iterator = DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)
        valid_iterator = DataLoader(self.valid_data, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
        return train_iterator, valid_iterator


if __name__ == "__main__":
    tokenize_en = get_tokenizer('spacy', language='en_core_web_sm')
    tokenize_de = get_tokenizer('spacy', language='de_core_news_sm')

    # Create a DataLoaderWrapper instance
    data_loader = Multi30kDataset(ext=('de', 'en'), 
                                  tokenize_en=tokenize_en, 
                                  tokenize_de=tokenize_de, 
                                  sos_token='<sos>', 
                                  eos_token='<eos>', 
                                  batch_size=32,
                                  batch_first=True, 
                                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Get iterators
    train_iter, valid_iter = data_loader.make_iter()
    for src, trg in train_iter:
        print(src.shape, trg.shape)
        break