import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import Multi30k, multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

class DataGenerator:
    UNK_IDX = 0
    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, split, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.token_transform, self.text_transform, self.vocab_transform = {}, {}, {}

        self.token_transform[source_lang] = get_tokenizer('spacy', language='de_core_news_sm')
        self.token_transform[target_lang] = get_tokenizer('spacy', language='en_core_web_sm')

        for ln in [self.source_lang, self.target_lang]:
            train_iter = Multi30k(split=split, language_pair=(self.source_lang, self.target_lang))
            self.vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(train_iter, ln),
                                                                 min_freq=1,
                                                                 specials=self.SPECIAL_SYMBOLS,
                                                                 special_first=True)

        for ln in [self.source_lang, self.target_lang]:
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)


        for ln in [self.source_lang, self.target_lang]:
            self.text_transform[ln] = self.sequential_transforms(self.token_transform[ln], # 토큰화(Tokenization)
                                                                 self.vocab_transform[ln], # 수치화(Numericalization)
                                                                 self.tensor_transform) # BOS/EOS를 추가하고 텐서를 생성


    def yield_tokens(self, data_iter, language: str):
        language_index = {self.source_lang: 0, self.target_lang: 1}

        for data_sample in data_iter:
            yield self.token_transform[language](data_sample[language_index[language]])


    def src_voc_size(self):
        # return len(self.vocab_transform[self.source_lang].get_itos())
        return len(self.vocab_transform[self.source_lang])


    def tgt_voc_size(self):
        # return len(self.vocab_transform[self.target_lang].get_itos())
        return len(self.vocab_transform[self.target_lang])


    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func


    def tensor_transform(self, token_ids):
        return torch.cat((torch.tensor([self.SOS_IDX]), torch.tensor(token_ids), torch.tensor([self.EOS_IDX])))
    

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.source_lang](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[self.target_lang](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)

        return src_batch, tgt_batch


# if __name__ == "__main__":
#     src_lang = 'en'
#     tgt_lang = 'de'
#     generator = DataGenerator(src_lang, tgt_lang)
#     train_iter = Multi30k(split='train', language_pair=(src_lang, tgt_lang))
#     train_dataloader = DataLoader(train_iter, batch_size=8, collate_fn=generator.collate_fn)

#     for data in train_dataloader:
#         src, tgt = data
#         print(src.shape, tgt.shape)