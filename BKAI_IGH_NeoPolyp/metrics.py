import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def soft_dice_score(output, target, smooth = 0.0, eps = 1e-7, dims=None):
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

    return dice_score


class DiceLoss(_Loss):
    def __init__(self, classes, log_loss, from_logits, ignore_index, smooth=0.0, eps=1e-7):
        super(DiceLoss, self).__init__()

        classes = torch.tensor(classes, dtype=torch.long)
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred, y_true):
        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)

            y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
            y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)

        else:
            y_true = F.one_hot(y_true.long(), num_classes)
            y_true = y_true.permute(0, 2, 1)

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()