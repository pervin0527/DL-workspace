import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torchvision import transforms

def get_img_and_bboxes():
    ## original images and bboxes
    image = cv2.imread("./zebras.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)

    bboxes = np.array([[223, 782, 623, 1074],
                       [597, 695, 1038, 1050],
                       [1088, 699, 1452, 1057],
                       [1544, 771, 1914, 1063]])
    labels = np.array([1, 1, 1, 1])

    img_with_bboxes = np.copy(image)
    for i in range(len(bboxes)):
        cv2.rectangle(img_with_bboxes, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]), color=(0, 255, 0), thickness=10)
    
    ## resize image & bboxes
    resized_image = cv2.resize(image, dsize=(800, 800), interpolation=cv2.INTER_CUBIC)
    h_ratio = 800 / image.shape[0]
    w_ratio = 800 / image.shape[1]
    # print(h_ratio, w_ratio)

    resized_bboxes = []
    ratioList = [w_ratio, h_ratio, w_ratio, h_ratio]

    for box in bboxes:
        box = [int(a*b) for a, b in zip(box, ratioList)]
        resized_bboxes.append(box)
    
    resized_bboxes = np.array(resized_bboxes)
    print(resized_bboxes)

    result_image = np.copy(resized_image)
    for i in range(len(resized_bboxes)):
        cv2.rectangle(result_image, (resized_bboxes[i][0], resized_bboxes[i][1]), (resized_bboxes[i][2], resized_bboxes[i][3]), color=(0, 255, 0), thickness=5)
    plt.imshow(result_image)
    plt.show()

    return resized_image, resized_bboxes


def build_backbone():
    model = torchvision.models.vgg16(pretrained=True).to(device)
    features = list(model.features)

    req_features = []
    sample = torch.zeros((1, 3, 800, 800)).float()
    output = sample.clone().to(device)
    for feature in features:
        output = feature(output)
        if output.size()[2] < 800 // 16:
            break

        req_features.append(feature)
        out_channels = output.size()[1]

    faster_rcnn_feature_extractor = nn.Sequential(*req_features)
    print(faster_rcnn_feature_extractor)

    return faster_rcnn_feature_extractor


def get_feature_map(image):
    transform = transforms.Compose([transforms.ToTensor()])
    imgTensor = transform(image).to(device)
    imgTensor = imgTensor.unsqueeze(0)
    output_map = backbone(imgTensor)

    print(output_map.size())

    imgArray = output_map.data.cpu().numpy().squeeze(0)
    fig = plt.figure(figsize=(12, 4))
    figNo = 1

    for i in range(5):
        fig.add_subplot(1, 5, figNo)
        plt.imshow(imgArray[i], cmap='gray')
        figNo += 1
        
    plt.show()

    return output_map

def generate_anchors(image):
    # sub-sampling rate = 1/16
    # image size : 800x800
    # sub-sampled feature map size : 800 x 1/16 = 50
    # 50 x 50 = 2500 anchors and each anchor generate 9 anchor boxes
    # total anchor boxes = 50 x 50 x 9 = 22500
    # x,y intervals to generate anchor box center

    feature_size = 800 // 16
    ctr_x = np.arange(16, (feature_size + 1) * 16, 16)
    ctr_y = np.arange(16, (feature_size + 1) * 16, 16)
    print(len(ctr_x))
    print(ctr_x)

    # coordinates of the 255 center points to generate anchor boxes
    index = 0
    ctr = np.zeros((2500, 2))

    for i in range(len(ctr_x)):
        for j in range(len(ctr_y)):
            ctr[index, 1] = ctr_x[i] - 8
            ctr[index, 0] = ctr_y[j] - 8
            index += 1

    # ctr => [[center x, center y], ...]
    print(ctr.shape)
    print(ctr)

    # display the 2500 anchors within image
    img_clone2 = np.copy(image)
    ctr_int = ctr.astype("int32")

    plt.figure(figsize=(7, 7))
    for i in range(ctr.shape[0]):
        cv2.circle(img_clone2, (ctr_int[i][0], ctr_int[i][1]),
                radius=1, color=(255, 0, 0), thickness=3)
    plt.imshow(img_clone2)
    plt.show()

    return ctr_int

def generate_anchor_boxes(feature_map, anchors, image, bboxes):
    # for each of the 2500 anchors, generate 9 anchor boxes
    # 2500 x 9 = 22500 anchor boxes
    ratios = [0.5, 1, 2]
    scales = [8, 16, 32]
    sub_sample = 16

    feature_size = feature_map.shape[2]
    anchor_boxes = np.zeros(((feature_size * feature_size * 9), 4))
    index = 0

    for c in anchors: # per anchors
        ctr_y, ctr_x = c
        for i in range(len(ratios)):     # per ratios
            for j in range(len(scales)): # per scales
                
                # anchor box height, width
                h = sub_sample * scales[j] * np.sqrt(ratios[i])
                w = sub_sample * scales[j] * np.sqrt(1./ ratios[i])
                
                # anchor box [x1, y1, x2, y2]
                anchor_boxes[index, 1] = ctr_y - h / 2.
                anchor_boxes[index, 0] = ctr_x - w / 2.
                anchor_boxes[index, 3] = ctr_y + h / 2.
                anchor_boxes[index, 2] = ctr_x + w / 2.
                index += 1
                
    print(anchor_boxes.shape)
    print(anchor_boxes)

    # display the anchor boxes of one anchor and the ground truth boxes
    img_clone = np.copy(image)

    # draw random anchor boxes
    for i in range(11025, 11034):
        x1 = int(anchor_boxes[i][0])
        y1 = int(anchor_boxes[i][1])
        x2 = int(anchor_boxes[i][2])
        y2 = int(anchor_boxes[i][3])
        
        cv2.rectangle(img_clone, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

    # draw ground truth boxes
    for i in range(len(bboxes)):
        cv2.rectangle(img_clone, (bboxes[i][0], bboxes[i][1]), (bboxes[i][2], bboxes[i][3]), color=(0, 255, 0), thickness=3)

    plt.imshow(img_clone)
    plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(device, torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print(device)

    image, bboxes = get_img_and_bboxes()
    backbone = build_backbone()
    feature_map = get_feature_map(image)
    print(feature_map)
    anchors = generate_anchors(image)
    anchor_boxes = generate_anchor_boxes(feature_map, anchors, image, bboxes)