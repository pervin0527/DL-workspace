import torch
from torch import nn
from torch.nn import functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.eps = 1e-7

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true_one_hot, dim=(2, 3))
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true_one_hot, dim=(2, 3))

        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1. - dice_coeff.mean()

        return dice_loss
    

class WeightDiceLoss(nn.Module):
    def __init__(self, weight=[1, 1, 1], eps=1e-7):
        super(WeightDiceLoss, self).__init__()
        self.weight = torch.tensor(weight).float()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true_one_hot, dim=(2, 3))
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true_one_hot, dim=(2, 3))

        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)

        if self.weight != None:
            self.weight = self.weight.to(dice_coeff.device)
        
        weighted_dice_coeff = dice_coeff * self.weight
        dice_loss = 1. - weighted_dice_coeff.mean()

        return dice_loss
    

class DiceCELoss(nn.Module):
    def __init__(self, weight=[1, 1, 1], eps=1e-7):
        super(DiceCELoss, self).__init__()
        self.eps = eps
        self.weight = weight

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true_one_hot, dim=(2, 3))
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true_one_hot, dim=(2, 3))

        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1. - dice_coeff.mean()

        if self.weight != None:
            self.weight = self.weight.to(dice_coeff.device)

        CE = F.cross_entropy(y_pred, y_true, weight=self.weight, reduction='mean')
        DICE_CE = CE + dice_loss

        return DICE_CE