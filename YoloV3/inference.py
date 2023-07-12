import cv2
import torch
import numpy as np
from model import YoloV3

from config import test_cfg
from utils import non_max_suppression, rescale_boxes_original

if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(DEVICE, torch.cuda.get_device_name(0))
    else:
        DEVICE = torch.device("cpu")
        print(DEVICE)

    font = cv2.FONT_HERSHEY_SIMPLEX
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", 
               "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    
    image = cv2.imread("./000001.jpg")
    x = cv2.resize(image, (test_cfg.img_size, test_cfg.img_size))
    x = np.expand_dims(np.transpose(x, (2, 0, 1)), 0)
    x = torch.tensor(x, dtype=torch.float32)
    print(x.shape)
    print(image.shape)
    
    model = YoloV3(test_cfg.img_size, len(classes)).to(DEVICE)
    if test_cfg.weight_path:
        model.load_state_dict(torch.load(test_cfg.weight_path))
    else:
        model.load_darknet_weights(test_cfg.darknet_weight_path)

    with torch.no_grad():
        x = x.to(DEVICE)
        prediction = model(x)
        prediction = non_max_suppression(prediction, test_cfg.conf_thres, test_cfg.nms_thres)
    
    if prediction is not None:
        prediction = rescale_boxes_original(prediction, test_cfg.img_size, image.shape[:2])
        for x1, y1, x2, y2, obj_conf, cls_conf, cls_pred in prediction:
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)

            text = f'{classes[int(cls_pred)]} : {obj_conf.item() * 100}'
            cv2.putText(image, text, (x1, y1), font, 1, (0, 0, 255), 2)
        
        cv2.imwrite("result.jpg", image)