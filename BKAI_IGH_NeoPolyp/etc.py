# class WeightedMultiClassTverskyLoss(nn.Module):
#     def __init__(self, class_weights=None, alpha=0.5, beta=0.5):
#         super(WeightedMultiClassTverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.class_weights = class_weights if class_weights is not None else torch.ones(3)  # 예시로 3개 클래스라고 가정

#     def forward(self, inputs, targets, smooth=1):
#         # comment out if your model contains a softmax or equivalent activation layer
#         inputs = F.softmax(inputs, dim=1) 

#         # One-hot encoding of true labels
#         targets_one_hot = torch.zeros_like(inputs)
#         targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
        
#         # Flatten tensors
#         inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, inputs.shape[1])
#         targets_one_hot = targets_one_hot.permute(0, 2, 3, 1).contiguous().view(-1, targets_one_hot.shape[1])
        
#         loss = 0
#         for i in range(inputs.shape[1]):  # Loop over classes
#             input_class = inputs[:, i]
#             true_class = targets_one_hot[:, i]
            
#             # True Positives, False Positives & False Negatives
#             TP = (input_class * true_class).sum()    
#             FP = ((1 - true_class) * input_class).sum()
#             FN = (true_class * (1 - input_class)).sum()

#             Tversky_class = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)
#             loss += self.class_weights[i] * (1 - Tversky_class)  # 클래스 가중치 적용
        
#         return loss / inputs.shape[1]  # Return average loss over classes


# class MultiClassTverskyLoss(nn.Module):
#     def __init__(self, weight=None, alpha=0.5, beta=0.5):
#         super(MultiClassTverskyLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta

#     def forward(self, y_pred, y_true, smooth=1):
#         # comment out if your model contains a softmax or equivalent activation layer
#         y_pred = F.softmax(y_pred, dim=1) 

#         # One-hot encoding of true labels
#         targets_one_hot = torch.zeros_like(y_pred)
#         targets_one_hot.scatter_(1, y_true.unsqueeze(1).long(), 1)
        
#         # Flatten tensors
#         y_pred = y_pred.permute(0, 2, 3, 1).contiguous().view(-1, y_pred.shape[1])
#         targets_one_hot = targets_one_hot.permute(0, 2, 3, 1).contiguous().view(-1, targets_one_hot.shape[1])
        
#         loss = 0
#         for i in range(y_pred.shape[1]):  # Loop over classes
#             y_pred_class = y_pred[:, i]
#             y_true_class = targets_one_hot[:, i]
            
#             # True Positives, False Positives & False Negatives
#             TP = (y_pred_class * y_true_class).sum()    
#             FP = ((1 - y_true_class) * y_pred_class).sum()
#             FN = (y_true_class * (1 - y_pred_class)).sum()

#             Tversky_class = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)
#             loss += 1 - Tversky_class
        
#         return loss / y_pred.shape[1]  # Return average loss over classes