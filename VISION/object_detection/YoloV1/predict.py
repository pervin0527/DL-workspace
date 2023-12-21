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
        if prob > threshold:
            x_center = x * img_width
            y_center = y * img_height
            width = w * img_width
            height = h * img_height
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            boxes.append([class_id, prob, x1, y1, width, height])

    return non_max_suppression(boxes, iou_threshold, threshold)


def predict_single_image(model, image, device="cuda"):
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        predictions = model(image.unsqueeze(0))

    bboxes = cellboxes_to_boxes(predictions, S=7)

    return bboxes[0]


def main():
    img_size = 448
    img_path = "./dog.jpg"
    weight_path = "/home/pervinco/Models/yolov1/2023-12-20_23-39-23/weights/ep_292_0.4649.pth"
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 이미지를 불러오고 원본 크기를 저장
    original_image = Image.open(img_path).convert('RGB')
    original_width, original_height = original_image.size

    # 모델에 입력하기 위해 이미지 변환
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    image = transform(original_image)

    model = Yolov1(grid_size=7, num_boxes=2, num_classes=20, pretrained=None).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    bboxes = predict_single_image(model, image, device=device)
    bboxes = post_processing(bboxes)

    # 예측된 상자 좌표를 원본 이미지 크기에 맞게 조정
    for box in bboxes:
        box[2] = (box[2] / img_size) * original_width
        box[3] = (box[3] / img_size) * original_height
        box[4] = (box[4] / img_size) * original_width
        box[5] = (box[5] / img_size) * original_height

    # 원본 이미지를 사용하여 경계 상자 그리기
    image_np = np.array(original_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for box in bboxes:
        class_id, prob, x1, y1, width, height = box
        x2 = x1 + width
        y2 = y1 + height
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_np, f"{classes[int(class_id)]}: {prob:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("result.jpg", image_np)

if __name__ == "__main__":
    main()
