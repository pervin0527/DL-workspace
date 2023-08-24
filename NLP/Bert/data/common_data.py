import os
import wget
import json
import argparse
import pandas as pd

from tqdm import tqdm
from vocab import load_vocab, build_corpus


def prepare_pretrain(args, vocab, outfile):
    """ pretrain data 준비 """
    line_cnt = 0
    with open(args.corpus, "r") as f:
        for line in f:
            line_cnt += 1

    # 단락 단위로 doc 생성 (vocab 적용)
    docs = []
    with open(args.corpus, "r") as f:
        doc = []
        for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"{args.corpus} loading", unit=" lines")):
            line = line.strip()
            if line == "":
                if doc:
                    docs.append(doc)
                    doc = []
            else:
                pieces = vocab.encode_as_pieces(line)
                doc.append(pieces)
        if doc:
            docs.append(doc)

    # 단락 단위로 json 형태로 저장
    with open(outfile, "w") as f:
        with tqdm(total=len(docs), desc=f"{outfile} saving") as pbar:
            for doc in docs:
                instance = {"doc": doc}
                f.write(json.dumps(instance))
                f.write("\n")
                pbar.update(1)


def prepare_train(args, vocab, infile, outfile):
    """ train data 준비 """
    df = pd.read_csv(infile, sep="\t", engine="python")
    with open(outfile, "w") as f:
        for index, row in df.iterrows():
            document = row["document"]
            if type(document) != str:
                continue
            instance = {"id": row["id"], "doc": vocab.encode_as_pieces(document), "label": row["label"]}
            f.write(json.dumps(instance))
            f.write("\n")
            print(f"build {outfile} {index + 1} / {len(df)}", end="\r")


def download_data(args):
    """ 데이터 다운로드 """
    print("download kowiki/ratings_train.txt")
    filename = wget.download("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", "kowiki")
    print()
    print("download kowiki/ratings_test.txt")
    filename = wget.download("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", "kowiki")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="prepare", type=str, required=False, help="동작모드 입니다. download: 학습할 데이터 다운로드, prepare: 학습할 데이터셋 생성")
    args = parser.parse_args()

    if args.mode == "download":
        download_data(args)
        
    elif args.mode == "prepare":
        vocab = load_vocab("kowiki.model")
        args.corpus = "kowiki/kowiki.txt"
        if not os.path.isfile(args.corpus):
            build_corpus("kowiki/kowiki.csv", args.corpus)
        if not os.path.isfile("kowiki/kowiki.json"):
            prepare_pretrain(args, vocab, "kowiki/kowiki.json")
        if not os.path.isfile("kowiki/ratings_train.json"):
            prepare_train(args, vocab, "kowiki/ratings_train.txt", "kowiki/ratings_train.json")
        if not os.path.isfile("kowiki/ratings_test.json"):
            prepare_train(args, vocab, "kowiki/ratings_test.txt", "kowiki/ratings_test.json")
    else:
        print(f"지원하지 않는 모드 입니다. {args.mode}\n- downlaod: 학습할 데이터 다운로드\n- preapre: 학습할 데이터셋 생성")
        exit(1)
