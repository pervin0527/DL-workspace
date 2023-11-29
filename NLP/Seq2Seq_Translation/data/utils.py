import os
import random
import warnings
import pandas as pd

from glob import glob

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def split_data(list1, list2, train_frac=0.7, valid_frac=0.15):
    assert len(list1) == len(list2)

    combined = list(zip(list1, list2))
    random.shuffle(combined)

    list1, list2 = zip(*combined)

    total_size = len(list1)
    train_size = int(total_size * train_frac)
    valid_size = int(total_size * valid_frac)

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