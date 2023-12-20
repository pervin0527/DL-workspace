import cv2
import torch

from PIL import Image
from torchvision.transforms import transforms

from models.yolov1 import Yolov1
from utils.detection_utils import cellboxes_to_boxes


def predict(model, image):
    with torch.no_grad():
        predictions = model(image)
    
    bboxes = cellboxes_to_boxes(predictions)
    print((bboxes))


def main():
    img_size = 448
    img_path = "./dog.jpg"
    weight_path = "/home/pervinco/Models/yolov1/2023-12-19_14-13-45/weights/ep_139_0.3750.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    model = Yolov1(grid_size=7, num_boxes=2, num_classes=20, pretrained=None).to(device)
    predict(model, image)


if __name__ == "__main__":
    main()