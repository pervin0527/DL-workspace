import os
from data.dataset import download_opensubtitles, read_file

if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/open_subtitles"
    download_opensubtitles(data_dir)

    data_dir = f"{data_dir}/ko-en"
    english_file = os.path.join(data_dir, "OpenSubtitles.en-ko.en")
    korean_file = os.path.join(data_dir, "OpenSubtitles.en-ko.ko")
    ids_file = os.path.join(data_dir, "OpenSubtitles.en-ko.ids")

    en_dataset = read_file(english_file)
    ko_dataset = read_file(korean_file)
    ids = read_file(ids_file)
    
    print(len(en_dataset), len(ko_dataset))
    for idx, (data1, data2) in enumerate(zip(en_dataset, ko_dataset)):
        print(data1)
        print(data2)

        if idx == 10:
            break