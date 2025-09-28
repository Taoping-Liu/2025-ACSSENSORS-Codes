import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from losses.base import ClassLoss

class SigmoidFocalLoss(ClassLoss):
    # https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", loss_term_weight=1.0):
        super(SigmoidFocalLoss, self).__init__(loss_term_weight)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,):
        inputs = inputs.float()
        one_hot_labels = torch.zeros_like(inputs)
        one_hot_labels.scatter_(1, targets.unsqueeze(1), 1) # shape: [batch_size, num_classes]
        targets = one_hot_labels.float()
        p = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean() 
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss * self.loss_term_weight
    

class SoftmaxFocalLoss(ClassLoss):
    # codes from https://github.com/pytorch/vision/issues/3250
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', loss_term_weight=1.0):
        super(SoftmaxFocalLoss, self).__init__(loss_term_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        loss = focal_loss
        if self.reduction == 'mean':
            loss =  focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum() 
        
        return loss * self.loss_term_weight

if __name__ == '__main__':
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,)) # shape (4,)
    one_hot_targets = F.one_hot(targets, num_classes=5)
    feats = None
    loss_sigmoid = SigmoidFocalLoss()(logits, feats, one_hot_targets)
    print(loss_sigmoid)
    
    reduction = 'mean'
    
    loss_softmax = SoftmaxFocalLoss(reduction=reduction)(logits, feats, targets)
    print(loss_softmax)
    

