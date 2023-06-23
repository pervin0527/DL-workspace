import torch
from torch import nn
from torchvision.models import vgg16
from config import cfg

def make_vgg16():
    # the 30th layer of features is relu of conv5_3
    model = vgg16()

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not cfg.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier

if __name__ == "__main__":
    test = make_vgg16()
    print(test)