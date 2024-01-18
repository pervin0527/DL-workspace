from torch import nn

class Transformer(nn.Module):
    def __init__(self, config, img_size, visualize):
        super(Transformer, self).__init__()
        

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, zero_head=False, visualize=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, visualize)