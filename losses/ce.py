
import torch.nn as nn
from losses.base import ClassLoss


class CELoss(ClassLoss):
    def __init__(self, label_smooth=True, eps=0.1, loss_term_weight=1.0):
        super(CELoss, self).__init__(loss_term_weight)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=eps) if label_smooth else nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss * self.loss_term_weight