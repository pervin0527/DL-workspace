import torch
from torch import nn
from torch.nn.modules.utils import _pair

from models.resnet import ResNetV2

class PatchEmbedding(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(PatchEmbedding, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None: ## grid는 conv의 feature map을 몇 개의 부분으로 분할할지 나타낸다.
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"]) ## patch size
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])