import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, n_classes, weight_ce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.n_classes = n_classes
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, input, target):
        smooth = 1e-5
        input = torch.softmax(input, dim=1)
        target = target.long()
        
        # One-hot encoding
        target_onehot = torch.zeros_like(input)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        intersect = torch.sum(input * target_onehot, dim=(2,3))
        denominator = torch.sum(input + target_onehot, dim=(2,3))
        
        dice_per_class = (2. * intersect + smooth) / (denominator + smooth)
        dice_loss = 1. - dice_per_class.mean()

        return dice_loss

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets.long())
        dice = self.dice_loss(inputs, targets)
        
        combined_loss = self.weight_ce * ce + self.weight_dice * dice
        return combined_loss