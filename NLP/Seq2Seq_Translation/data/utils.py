import os
import random
import warnings
import pandas as pd

from glob import glob
from torchtext.vocab import build_vocab_from_iterator

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def split_data(list1, list2, train_frac=0.7, valid_frac=0.15):
    # 데이터 길이 확인
    assert len(list1) == len(list2) == 1402407

    # 데이터 쌍을 만듦 (연관된 데이터 유지)
    combined = list(zip(list1, list2))

    # 데이터 섞기
    random.shuffle(combined)

    # 다시 분할
    list1, list2 = zip(*combined)

    # 각 세트의 크기 계산
    total_size = len(list1)
    train_size = int(total_size * train_frac)
    valid_size = int(total_size * valid_frac)

    # 데이터를 train, valid, test로 분할
    train_data = (list1[:train_size], list2[:train_size])
    valid_data = (list1[train_size:train_size+valid_size], list2[train_size:train_size+valid_size])
    test_data = (list1[train_size+valid_size:], list2[train_size+valid_size:])

    return train_data, valid_data, test_data


def get_total_data(data_dir, reverse=False):
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

    if not reverse:
        total_data = [df["원문"].tolist(), df["번역문"].tolist()]
    else:
        total_data = [df["번역문"].tolist(), df["원문"].tolist()]

    return total_data


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