import cv2
import torch
import numpy as np

from PIL import Image
from torchvision.transforms import transforms

from models.yolov1 import Yolov1
from utils.detection_utils import cellboxes_to_boxes, non_max_suppression

def post_processing(bboxes, threshold=0.5, iou_threshold=0.45, img_width=448, img_height=448):
    boxes = []
    for pred in bboxes:
        class_id, prob, x, y, w, h = pred
        if prob > threshold:  # threshold는 설정해야 하는 신뢰도 임계값
            x_center = int(x * img_width)
            y_center = int(y * img_height)
            width = int(w * img_width)
            height = int(h * img_height)
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            boxes.append([class_id, prob, x1, y1, width, height])

    return non_max_suppression(boxes, iou_threshold, threshold)


def predict_single_image(model, image, device="cuda"):
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        predictions = model(image.unsqueeze(0))  # 단일 이미지 처리를 위해 차원 추가

    # 예측 결과를 일반적인 박스 형태로 변환
    bboxes = cellboxes_to_boxes(predictions, S=7)
    return bboxes[0]  # 첫 번째 (그리고 유일한) 이미지의 박스 반환


def main():
    img_size = 448
    img_path = "./dog.jpg"
    weight_path = "/home/pervinco/Models/yolov1/2023-12-19_14-13-45/weights/ep_139_0.3750.pth"
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    x = transform(image)

    model = Yolov1(grid_size=7, num_boxes=2, num_classes=20, pretrained=None).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    bboxes = predict_single_image(model, x, device=device)
    bboxes = post_processing(bboxes)
    print(bboxes)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    for box in bboxes:
        class_id, prob, x1, y1, width, height = box
        x2 = x1 + width
        y2 = y1 + height
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{classes[int(class_id)]}: {prob:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("result.jpg", image)

if __name__ == "__main__":
    main()