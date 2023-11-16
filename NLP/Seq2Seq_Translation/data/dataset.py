import os
import zipfile
import requests

def download_file(url, target_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())


def download_opensubtitles(save_dir):
    url = "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-ko.txt.zip"
    zip_file = "ko-en.zip"
    zip_out_dir = "ko-en"

    if not os.path.isdir(save_dir) and not os.path.isfile(f"{save_dir}/{zip_file}"):
        print("Download Dataset.")
        os.makedirs(save_dir)

        download_file(url, f"/{save_dir}/{zip_file}")
        with zipfile.ZipFile(f"/{save_dir}/{zip_file}", 'r') as zip_ref:
            zip_ref.extractall(f"/{save_dir}/{zip_out_dir}")

        print("Done")

    else:
        print("Dataset already exist.")


def read_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    return lines