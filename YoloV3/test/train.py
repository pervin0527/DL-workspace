import torch
from config import train_opt
from utils.general import make_dirs
from models.yolo import Model

def train():
    if torch.cuda.is_available():
        DEVICE = torch.device(train_opt.device)
        print(DEVICE, torch.cuda.get_device_name(0))
    else:
        DEVICE = torch.device("cpu")
        print(DEVICE)

    w = f"{train_opt.save_dir}/weights"
    make_dirs(w)
    last, best = f"{w}/last.pt", f"{w}/best.pt"

    hyp = train_opt.hyp
    train_path, val_path, names = train_opt.data["train"], train_opt.data["val"], train_opt.data["classes"]

    weights = train_opt.weight_dir
    if weights != None and weights != "":
        ckpt = torch.load(weights, map_location='cpu')
        print("Pretrained Weight Loaded.")
    model = Model(train_opt.model, ch=3, nc=train_opt.num_classes, anchors=train_opt.anchors).to(DEVICE)  # create
    print(model)



if __name__ == "__main__":
    train()