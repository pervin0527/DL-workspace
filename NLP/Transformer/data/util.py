import os
import wget
import pickle

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"{path} folder maded")
    else:
        print(f"{path} is already exist.")


def download_multi30k(save_path):
    URL = "https://github.com/multi30k/dataset/raw/master/data/task1/raw"
    FILES = ["test_2016_flickr.de.gz",
             "test_2016_flickr.en.gz",
             "train.de.gz",
             "train.en.gz",
             "val.de.gz",
             "val.en.gz"]
    
    save_path = f"{save_path}/Multi30k"
    make_dir(save_path)

    for file in FILES:
        file_name = file[:-3]
        if file_name == "test_2016_flickr.de_gz":
            file_name = "test.de"
        elif file_name == "test_2016_flickr.en.gz":
            file_name = "test.en"

        if os.path.exists(f"{save_path}/{file_name}"):
            pass
        else:
            url = f"{URL}/{file}"
            # print(f"{url}\n")

            wget.download(url, out=save_path)
            os.system(f"gzip -d {save_path}/{file}")
        
            if file == FILES[0]:
                os.system(f"cp {save_path}/{file[:-3]} {save_path}/test.de")
            elif file == FILES[1]:
                os.system(f"cp {save_path}/{file[:-3]} {save_path}/test.en")


def load_pickle(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, fname):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def make_cache(data_path):
    cache_path = f"{data_path}/cache"
    make_dir(cache_path)

    if not os.path.exists(f"{cache_path}/train.pkl"):
        for name in ["train", "val", "test"]:
            pkl_file_name = f"{cache_path}/{name}.pkl"

            with open(f"{data_path}/{name}.en", "r") as file:
                en = [text.rstrip() for text in file]
            
            with open(f"{data_path}/{name}.de", "r") as file:
                de = [text.rstrip() for text in file]
            
            data = [(en_text, de_text) for en_text, de_text in zip(en, de)]
            save_pickle(data, pkl_file_name)