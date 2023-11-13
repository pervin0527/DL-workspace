import os
import wget
import pickle

def data_download(data_dir):
    URL = "https://github.com/multi30k/dataset/raw/master/data/task1/raw"
    FILES = ["test_2016_flickr.de.gz",
             "test_2016_flickr.en.gz",
             "train.de.gz",
             "train.en.gz",
             "val.de.gz",
             "val.en.gz"]
    
    os.makedirs(data_dir)
    
    for file in FILES:
        tmp_url = f"{URL}/{file}"
        wget.download(tmp_url, data_dir)
        file_name = tmp_url.split('/')[-1]
        os.system(f"gzip -d {data_dir}/{file_name}")

    os.system(f"mv {data_dir}/test_2016_flickr.de {data_dir}/test.de")
    os.system(f"mv {data_dir}/test_2016_flickr.en {data_dir}/test.en")

def save_pkl(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_pkl(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data