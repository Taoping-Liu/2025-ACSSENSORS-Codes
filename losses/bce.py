import torch
import torch.nn as nn
from losses.base import ClassLoss


class BCELoss(ClassLoss):
    def __init__(self, loss_term_weight=1.0):
        super(BCELoss, self).__init__(loss_term_weight)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, labels):
        one_hot_labels = torch.zeros_like(logits)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1) # shape: [batch_size, num_classes]
        loss = self.criterion(logits, one_hot_labels)

        return loss * self.loss_term_weight