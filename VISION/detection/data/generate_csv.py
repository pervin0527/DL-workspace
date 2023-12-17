import csv

def read_txt(txt_path):
    if isinstance(txt_path, list):
        total_lines = []
        for txt in txt_path:
            line = open(txt, "r").read().strip().split('\n')
            total_lines.extend(line)
    else:
        total_lines = open(txt, "r").read().strip().split('\n')
    
    return total_lines


def write_csv(file_list, save_dir):
    with open(save_dir, mode="w", newline="") as csv_file:
        for f in file_list:
            image_file = f.replace("\n", "")
            label_file = image_file.replace("JPEGImages", "labels").replace(".jpg", ".txt")
            data = [image_file, label_file]

            writer = csv.writer(csv_file)
            writer.writerow(data)


if __name__ == "__main__":
    train_text_files = ["./2007_train.txt", "./2007_val.txt", "./2012_train.txt", "./2012_val.txt"]
    test_text_file = ["./2007_test.txt"]

    train_files = read_txt(train_text_files)
    test_files = read_txt(test_text_file)

    write_csv(train_files, "./train.csv")
    write_csv(test_files, "./test.csv")