import os
import warnings
import pandas as pd

from glob import glob
from torchtext.vocab import build_vocab_from_iterator

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def get_total_data(data_dir):
    if os.path.isfile(f"{data_dir}/total_data.csv"):
        print("total_data.csv exist.")
        df = pd.read_csv(f"{data_dir}/total_data.csv")
    else:
        print("total_data.csv not exist.")
        df = pd.DataFrame(columns = ['원문','번역문'])
        files = sorted(glob(f"{data_dir}/*.xlsx"))

        for file in files:
            print(file)
            tmp = pd.read_excel(file)
            df = pd.concat([df, tmp[['원문','번역문']]])

        df.to_csv(f'{data_dir}/total_data.csv', index=False, encoding='utf-8-sig')
        print("total_data.csv generated.")

    return df


def insert_tokens(arr):
    arr.insert(0, "<sos>")
    arr.append("<eos>")

    return arr


def tokenize(df, ko_tokenizer, en_tokenizer):
    total_ko_tokens = []
    total_en_tokens = []
    for index, row in df.iterrows():
        ko, en = row["원문"], row["번역문"]
        
        ko_tokens = ko_tokenizer.morphs(ko.strip())
        en_tokens = en_tokenizer(en.strip())

        ko_tokens = insert_tokens(ko_tokens)
        en_tokens = insert_tokens(en_tokens)

        total_ko_tokens.append(ko_tokens)
        total_en_tokens.append(en_tokens)

    return total_ko_tokens, total_en_tokens


def build_vocab(data_tokens):
    return build_vocab_from_iterator(data_tokens, specials=['<unk>', '<sos>', "<eos>"], min_freq=1)


def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]