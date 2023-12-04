import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import wget
from torchtext import transforms
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from utils import load_pkl, save_pkl

class Multi30k:
    UNK, UNK_IDX = "<unk>", 0
    PAD, PAD_IDX = "<pad>", 1
    SOS, SOS_IDX = "<sos>", 2
    EOS, EOS_IDX = "<eos>", 3
    SPECIALS = {UNK : UNK_IDX, PAD : PAD_IDX, SOS : SOS_IDX, EOS : EOS_IDX}

    URL = "https://github.com/multi30k/dataset/raw/master/data/task1/raw"
    FILES = ["test_2016_flickr.de.gz",
             "test_2016_flickr.en.gz",
             "train.de.gz",
             "train.en.gz",
             "val.de.gz",
             "val.en.gz"]

    def __init__(self, data_dir, source_language="en", target_language="de", max_seq_len=256, vocab_min_freq=2):
        self.data_dir = f"{data_dir}/Multi30k"
        self.cache_dir = f"{self.data_dir}/caches"
        
        if not os.path.isdir(self.data_dir):
            print("Data Download")
            self.download()

        self.source_language, self.target_language = source_language, target_language
        self.max_seq_len = max_seq_len
        self.vocab_min_freq = vocab_min_freq

        self.source_tokenizer = self.build_tokenizer(self.source_language)
        self.target_tokenizer = self.build_tokenizer(self.target_language)

        self.train, self.valid, self.test = None, None, None
        self.build_dataset()

        self.src_vocab, trg_vocab = None, None
        self.build_vocab()

        self.src_transform, self.trg_transform = None, None
        self.build_transform()

        
    def download(self):
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"Data Dir {self.data_dir} maded.")
        
            for file in self.FILES:
                url = f"{self.URL}/{file}"
                print(url)

                wget.download(url, out=self.data_dir)
                file_name = url.split('/')[-1]
                os.system(f"gzip -d {self.data_dir}/{file_name}")


    def build_tokenizer(self, language):
        from torchtext.data.utils import get_tokenizer
        spacy_lang_dict = {'en': "en_core_web_sm",
                           'de': "de_core_news_sm"}
        
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
                train_en = [text.rstrip() for text in f]
            
            with open(f"{self.data_dir}/train.de") as f:
                train_de = [text.rstrip() for text in f]
            
            if self.source_language == "en":
                self.train = [(en, de) for en, de in zip(train_en, train_de)]
            else: 
                self.train = [(de, en) for de, en in zip(train_de, train_en)]
            save_pkl(self.train, train_pkl)

        valid_pkl = f"{self.cache_dir}/val.pkl"
        if os.path.exists(valid_pkl):
            self.valid = load_pkl(valid_pkl)
        else:
            with open(f"{self.data_dir}/val.en") as f:
                valid_en = [text.rstrip() for text in f]
            
            with open(f"{self.data_dir}/val.de") as f:
                valid_de = [text.rstrip() for text in f]
            
            if self.source_language == "en":
                self.valid = [(en, de) for en, de in zip(valid_en, valid_de)]
            else: 
                self.valid = [(de, en) for de, en in zip(valid_de, valid_en)]
            save_pkl(self.valid, valid_pkl)

        test_pkl = f"{self.cache_dir}/test.pkl"
        if os.path.exists(test_pkl):
            self.test = load_pkl(test_pkl)
        else:
            with open(f"{self.data_dir}/test_2016_flickr.en") as f:
                test_en = [text.rstrip() for text in f]
            
            with open(f"{self.data_dir}/test_2016_flickr.de") as f:
                test_de = [text.rstrip() for text in f]
            
            if self.source_language == "en":
                self.test = [(en, de) for en, de in zip(test_en, test_de)]
            else: 
                self.test = [(de, en) for de, en in zip(test_de, test_en)]
            save_pkl(self.test, test_pkl)


    def build_vocab(self):
        assert self.train is not None

        def yield_tokens(is_src=True):
            for text_pair in self.train:
                if is_src:
                    yield [str(token) for token in self.source_tokenizer(text_pair[0])]
                else:
                    yield [str(token) for token in self.target_tokenizer(text_pair[1])]

        src_vocab_file = f"{self.cache_dir}/vocab_{self.source_language}.pkl"
        if os.path.exists(src_vocab_file):
            src_vocab = load_pkl(src_vocab_file)
        else:
            src_vocab = build_vocab_from_iterator(yield_tokens(is_src=True), min_freq=self.vocab_min_freq, specials=self.SPECIALS.keys())
            src_vocab.set_default_index(self.UNK_IDX)
            save_pkl(src_vocab, src_vocab_file)

        trg_vocab_file = f"{self.cache_dir}/vocab_{self.target_language}.pkl"
        if os.path.exists(trg_vocab_file):
            trg_vocab = load_pkl(trg_vocab_file)
        else:
            trg_vocab = build_vocab_from_iterator(yield_tokens(is_src=False), min_freq=self.vocab_min_freq, specials=self.SPECIALS.keys())
            trg_vocab.set_default_index(self.UNK_IDX)
            save_pkl(trg_vocab, trg_vocab_file)

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab


    def build_transform(self):
        def get_transform(self, vocab):
            return transforms.Sequential(transforms.VocabTransform(vocab),
                                         transforms.Truncate(self.max_seq_len-2),
                                         transforms.AddToken(token=self.SOS_IDX, begin=True),
                                         transforms.AddToken(token=self.EOS_IDX, begin=False),
                                         transforms.ToTensor(padding_value=self.PAD_IDX))

        self.src_transform = get_transform(self, self.src_vocab)
        self.trg_transform = get_transform(self, self.trg_vocab)


    def collate_fn(self, pairs):
        src = [self.source_tokenizer(pair[0]) for pair in pairs]
        trg = [self.target_tokenizer(pair[1]) for pair in pairs]
        batch_src = self.src_transform(src)
        batch_trg = self.trg_transform(trg)

        return (batch_src, batch_trg)


    def get_iter(self, **kwargs):
        if self.src_transform is None:
            self.build_transform()

        train_iter = DataLoader(self.train, collate_fn=self.collate_fn, **kwargs)
        valid_iter = DataLoader(self.valid, collate_fn=self.collate_fn, **kwargs)
        test_iter = DataLoader(self.test, collate_fn=self.collate_fn, **kwargs)

        return train_iter, valid_iter, test_iter


    def translate(self, model, src_sentence: str, decode_func):
        model.eval()
        src = self.src_transform([self.source_tokenizer(src_sentence)]).view(1, -1)
        num_tokens = src.shape[1]
        trg_tokens = decode_func(model, src, max_len=num_tokens + 5, start_symbol=self.SOS_IDX, end_symbol=self.EOS_IDX).flatten().cpu().numpy()
        trg_sentence = " ".join(self.trg_vocab.lookup_tokens(trg_tokens))

        return trg_sentence
