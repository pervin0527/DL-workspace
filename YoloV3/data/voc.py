import os
import shutil
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_file_path, classes, output_path):
    with open(output_path, "w") as f:
        tree=ET.parse(xml_file_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def make_labels(path, classes_txt_path, set_type):
    save_path = f'{path}/yolo/{set_type}'
    if not os.path.isdir(save_path):
        os.makedirs(f'{save_path}/images')
        os.makedirs(f'{save_path}/labels')

    with open(classes_txt_path, "r") as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes]

    files = open(f'{path}/ImageSets/Main/{set_type}.txt').read().strip().split()
    for file in files:
        img_file_path = f'{path}/JPEGImages/{file}.jpg'
        xml_file_path = f'{path}/Annotations/{file}.xml'

        convert_annotation(xml_file_path, classes, f"{save_path}/labels/{file}.txt")
        shutil.copyfile(img_file_path, f'{save_path}/images/{file}.jpg')

if __name__ == "__main__":
    PATH = "/home/pervinco/Datasets/PASCAL_VOC/VOCdevkit/VOC2012"
    CLASSES = "/home/pervinco/DL-workspace/YoloV3/test/classes.txt"
    make_labels(PATH, CLASSES, "train")
    make_labels(PATH, CLASSES, "val")
