import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, weight=[1, 1, 4], crossentropy=False):
        super(DiceLoss, self).__init__()
        self.eps = 1e-7
        self.num_classes = num_classes
        self.crossentropy = crossentropy
        if crossentropy:
            self.ce = nn.CrossEntropyLoss()
        
        self.weight = weight

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true, dim=(2, 3))
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true, dim=(2, 3))

        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1. - dice_coeff.mean()

        if self.crossentropy:
            crossentropy_loss = self.ce(y_pred, y_true)
            total_loss = 0.4 * crossentropy_loss + 0.6 * dice_loss

            return total_loss

        else:
            return dice_loss