import os
import zipfile
import requests

def download_wikitext(save_dir):
    wikitext2_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip?ref=blog.salesforceairesearch.com"
    wikitext103_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip?ref=blog.salesforceairesearch.com"

    def download_file(url, filename):
        response = requests.get(url)
        with open(filename, 'wb') as file:
            file.write(response.content)

    def unzip_file(zip_filename, extract_to):
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    if not os.path.isdir(save_dir) and not os.path.isfile(f"{save_dir}/wikitext-2/wiki.train.tokens"):
        print("Download Dataset.")
        os.makedirs(save_dir)

        wikitext2_zip = os.path.join(save_dir, "wikitext-2-v1.zip")
        wikitext103_zip = os.path.join(save_dir, "wikitext-103-v1.zip")

        download_file(wikitext2_url, wikitext2_zip)
        unzip_file(wikitext2_zip, save_dir)

        download_file(wikitext103_url, wikitext103_zip)
        unzip_file(wikitext103_zip, save_dir)
        print("Done")

    else:
        print("Dataset already exist.")