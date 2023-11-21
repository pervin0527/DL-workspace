import os

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    return text