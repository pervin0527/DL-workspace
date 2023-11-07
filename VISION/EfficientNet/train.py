import torch
from models.model import EfficientNet

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 224, 224)
    model = EfficientNet.from_pretrained("efficientnet-b0")
    model.eval()
    outputs = model(inputs)
    print(outputs.shape)