# -*- coding: UTF-8 -*-

from __future__ import absolute_import
import torch.nn as nn
import torch
import torch.nn.functional as F

""" 
    Class that defines the Dice Loss function.
"""

class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        pred_probs = torch.sigmoid(y_pred)
        y_true_f = y_true.view(-1)
        y_pred_f = pred_probs.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def forward(self, y_pred, y_true):
        return 1 - self.dice_coef(y_pred, y_true)

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, weights):
        probs = F.sigmoid(y_pred)
        num = y_true.size(0)
        w = weights.view(num, -1)
        # w2 = w * w
        w2 = w
        m1 = probs.view(num, -1)
        m2 = y_true.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2 * intersection).sum(1) + 1) / (
            (w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


""" 
    Class that defines the Root Mean Square Loss function.
"""


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))



"""
    Class that defines the Binary Cross Entropy Loss Function
"""


class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        return self.bce(y_pred.view(batch_size, -1), y_true.view(batch_size, -1))


"""
    Class that defines the Weighted Binary Cross Entropy Loss Function
"""


class WBCELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(WBCELoss, self).__init__()
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true, weight):
        batch_size = y_pred.size(0)
        loss = self.bce(y_pred.view(batch_size, -1), y_true.view(batch_size, -1))
        if self.reduction == "mean":
            return torch.mean(loss * weight.view(batch_size, -1))
        elif self.reduction == "sum":
            return torch.sum(loss * weight.view(batch_size, -1))
        else:
            raise ValueError('reduction should one of sum or mean')


"""
    Class that defines the Cross Entropy Loss Function
"""


class CELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        class_num = y_pred.size(1)
        return self.bce(y_pred.view(batch_size, class_num, -1), y_true.view(batch_size, -1))



"""
    Class that defines the Cross Entropy Loss Function
"""

class WCELoss(nn.Module):
    def __init__(self):
        super(WCELoss, self).__init__()

    def forward(self, y_pred, y_true, weights):
        y_true = y_true / (y_true.sum(2).sum(2, dtype=torch.float).unsqueeze(-1).unsqueeze(-1))
        y_true[y_true != y_true] = 0.0
        y_true = torch.sum(y_true, dim=1, dtype=torch.float).unsqueeze(1)
        y_true = y_true * weights.to(torch.float)
        old_range = torch.max(y_true) - torch.min(y_true)
        new_range = 100 - 1
        y_true = (((y_true - torch.min(y_true)) * new_range) / old_range) + 1
        return -torch.mean(torch.sum(y_true * torch.log(F.softmax(y_pred, dim=1)), dim=1))


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.mse(inputs, targets)


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, inputs, targets):
        return self.mae(inputs, targets)