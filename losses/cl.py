import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.base import FeatureLoss


class ContrastiveLoss(FeatureLoss):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        # features: (N, D), labels: (N,)
        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(features, features, p=2)

        # Create positive and negative mask
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float() - torch.eye(labels.shape[0]).to(labels.device)
        positive_mask = torch.clamp(positive_mask, 0, 1)
        negative_mask = (labels != labels.t()).float()

        # Compute loss
        positive_loss = (dist_matrix * positive_mask).sum() / positive_mask.sum()
        negative_loss = (F.relu(self.margin - dist_matrix) * negative_mask).sum() / negative_mask.sum()

        return positive_loss + negative_loss