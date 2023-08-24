import os
import re
import wget
import json
import datetime
import pandas as pd

def list_wiki(dir):
    file_paths = []
    file_names = os.listdir(dir)
    for file_name in file_names:
        file_path = os.path.join(dir, file_name)

        if os.path.isdir(file_path):
            file_paths.extend(list_wiki(file_path))
        else:
            find = re.findall(r"wiki_[0-9][0-9]", file_path)
            if 0 < len(find):
                file_paths.append(file_path)

    return sorted(file_paths)

""" 여러줄띄기(\n\n...) 한줄띄기로(\n) 변경 """
def trim_text(line):
    data = json.loads(line)
    text = data["text"]
    value = list(filter(lambda x: len(x) > 0, text.split('\n')))
    data["text"] = "\n".join(value)
    return data


""" csv 파일을 제외한 나머지 파일 삭제 """
def del_garbage(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        filepath = os.path.join(dirname, filename)

        if os.path.isdir(filepath):
            del_garbage(filepath)
            os.rmdir(filepath)
        else:
            if not filename.endswith(".csv"):
                os.remove(filepath)


if __name__ == "__main__":
    url = "https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-meta-current.xml.bz2"
    data_dir = "./kowiki"
    sep = u"\u241D"

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(f"{data_dir}/kowiki_*.csv"):
        download_file = wget.download(url, data_dir) ## kowiki-latest-pages-meta-current.xml.bz2qout0xlm.tmp
        os.system(f"python3 wiki_extractor.py -o {data_dir} --json {download_file}") ## extract download_file

        # text 여러줄 띄기를 한줄 띄기로 합침
        dataset = []
        filenames = list_wiki(data_dir)
        for filename in filenames:
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        dataset.append(trim_text(line))
        
        # 자장파일 결정
        now = datetime.datetime.now().strftime("%Y%m%d")
        output = f"{data_dir}/kowiki_{now}.csv"

        # 위키저장 (csv)
        if 0 < len(dataset):
            df = pd.DataFrame(data=dataset)
            df.to_csv(output, sep=sep, index=False)
        
        # 파일삭제
        del_garbage(data_dir)