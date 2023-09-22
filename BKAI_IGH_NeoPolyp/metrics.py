import torch
import torch.nn as nn
import torch.nn.functional as F

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, y_pred, y_true):
#         predictions = F.softmax(y_pred, dim=1)
#         one_hot_gt = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()
        
#         predictions = predictions.float()
#         one_hot_gt = one_hot_gt.reshape(-1)
#         predictions = predictions.reshape(-1)
        
#         intersection = (predictions * one_hot_gt).sum()
#         union = predictions.sum() + one_hot_gt.sum()

#         dice = (2. * intersection + self.smooth) / (union + self.smooth)

#         return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true_one_hot, dim=(2, 3))
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true_one_hot, dim=(2, 3))

        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1. - dice_coeff.mean()

        return dice_loss
    

class DiceCELoss(nn.Module):
    def __init__(self, weight=None, eps=1e-7):
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

        CE = F.cross_entropy(y_pred, y_true, weight=self.weight, reduction='mean')
        DICE_CE = CE + dice_loss

        return DICE_CE