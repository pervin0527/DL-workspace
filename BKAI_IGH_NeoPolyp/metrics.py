import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, crossentropy=False, weight=None, give_penalty=False, penalty_factor=2.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.crossentropy = crossentropy
        if crossentropy:
            self.ce = nn.CrossEntropyLoss()
        self.weight = weight

        self.penalty = give_penalty
        if give_penalty:
            self.penalty_factor = penalty_factor


    def one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()


    def penalty_loss(self, y_pred, y_true):
        total_pixels = y_true.numel()
        
        mask_true_0 = (y_true == 0).unsqueeze(1).float()
        penalty_1 = torch.sum(mask_true_0 * (y_pred[:,1,:,:] + y_pred[:,2,:,:]))
        
        mask_true_1 = (y_true == 1).unsqueeze(1).float()
        penalty_2 = torch.sum(mask_true_1 * y_pred[:,2,:,:])

        mask_true_2 = (y_true == 2).unsqueeze(1).float()
        penalty_3 = torch.sum(mask_true_2 * y_pred[:,1,:,:])
        
        avg_penalty = (penalty_1 + penalty_2 + penalty_3) / total_pixels
        
        return avg_penalty * self.penalty_factor


    def dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss

        return loss


    def forward(self, y_pred, y_true):
        if self.weight == None:
            self.weight = [1] * self.num_classes

        y_pred = torch.softmax(y_pred, dim=1)
        y_true_one_hot = self.one_hot_encoder(y_true)

        assert y_pred.size() == y_true_one_hot.size(), 'predict {} & y_true {} shape do not match'.format(y_pred.size(), y_true_one_hot.size())

        loss = 0.0
        for i in range(0, self.num_classes):
            dice = self.dice_loss(y_pred[:, i, :, :], y_true_one_hot[:, i, :, :])
            loss += dice * self.weight[i]

        dice_loss = loss / self.num_classes
        
        if self.penalty:
            dice_loss += self.penalty_loss(y_pred, y_true)

        if self.crossentropy:
            cross_entropy_loss = self.ce(y_pred, y_true)
            final_loss = 0.4 * cross_entropy_loss + 0.6 * dice_loss
            return final_loss
        else:
            return dice_loss
