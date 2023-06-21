from torchvision import ops
from backbones.resnet import resnet_fpn
from configs.train_config import train_cfg
from utils.anchor_utils import AnchorsGenerator

def build_model(num_classes):
    backbone_name = train_cfg.BACKBONE
    print(backbone_name)
    anchor_sizes = tuple((f,) for f in train_cfg.ANCHOR_SIZE)
    aspect_ratios = tuple((f,) for f in train_cfg.ANCHOR_RATIO) * len(anchor_sizes)

    anchor_generator = AnchorsGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    if backbone_name == "resnet50":
        backbone = resnet_fpn(backbone_name, num_classes=num_classes)
        print(backbone)
