import os
import xml.etree.ElementTree as ET

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open(f'{data_dir}/VOC{year}/Annotations/{image_id}.xml')
    out_file = open(f'{data_dir}/VOC{year}/{folder_name}/{image_id}.txt', 'w')

    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == "__main__":
    data_dir = "/home/pervinco/Datasets/PASCAL_VOC/VOCDevkit"
    folder_name = "yolov1"
    
    for year, image_set in sets:
        if not os.path.isdir(f"{data_dir}/VOC{year}/{folder_name}") and not os.path.exists(f"{data_dir}/VOC{year}/{folder_name}/{year}"):
            os.makedirs(f"{data_dir}/VOC{year}/{folder_name}")
        
        image_ids = open(f"{data_dir}/VOC{year}/ImageSets/Main/{image_set}.txt").read().strip().split()
        
        if image_set in ['train', 'val']:
            file_name = "./train.txt"
        elif image_set == 'test':
            file_name = "./test.txt"
        else:
            raise ValueError("Invalid image set")

        with open(file_name, "a") as list_file:
            for image_id in image_ids:
                list_file.write(f"{data_dir}/VOC{year}/JPEGImages/{image_id}.jpg\n")
                convert_annotation(year, image_id)