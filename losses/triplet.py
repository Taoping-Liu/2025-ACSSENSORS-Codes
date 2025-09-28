import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.base import FeatureLoss


class TripletLoss(FeatureLoss):
    """

    Reference:
        Hermans et al. In Defense of the Triplet Loss for
            Person Re-Identification. arXiv:1703.07737.
    Imported from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/losses/triplet_loss.py. (hard mining)
                  https://github.com/ShiqiYu/OpenGait/blob/master/opengait/modeling/losses/triplet.py (no hard mining)
    Args:
        margin (float, optional): Margin for triplet loss. Defaults to 0.3.
        loss_term_weight (float, optional): Weight of the loss. Defaults to 1.0.
        hard_mining (bool, optional): Whether to perform hard mining.
            Defaults to True.
    """

    def __init__(self,
                 margin: float = 0.3,
                 loss_term_weight: float = 1.0,
                 hard_mining=True):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.hard_mining = hard_mining

    def compute_dist(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Compute Euclidean distance
        dist = torch.pow(x, 2).sum(
            dim=1, keepdim=True).expand(batch_size, batch_size)
        dist = dist + dist.t()
        dist.addmm_(x, x.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def hard_mining_triplet_loss_forward(
            self, inputs: torch.Tensor,
            targets: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (batch_size).

        Returns:
            torch.Tensor: triplet loss with hard mining.
        """
        batch_size = inputs.size(0)
        dist = self.compute_dist(inputs)

        # For each anchor, find the furthest positive sample
        # and nearest negative sample in the embedding space
        mask = targets.expand(batch_size, batch_size).eq(
            targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0)) # hardest positive
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0)) # hardest negative
        dist_ap = torch.cat(dist_ap) # [batch_size]
        dist_an = torch.cat(dist_an) # [batch_size]

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
    

    def no_hard_mining_triplet_loss_forward(
            self, inputs: torch.Tensor,
            targets: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (num_classes).

        Returns:
            torch.Tensor: triplet loss without hard mining.
        """
        batch_size = inputs.size(0)
        dist = self.compute_dist(inputs)
        # Create mask for positive and negative samples
        matches = (targets.unsqueeze(1) ==
                   targets.unsqueeze(0)).bool()  # [batch_size, batch_size]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        
        total_loss = 0
        for i in range(batch_size):
            dist_ap_i = dist[i][matches[i]].unsqueeze(1) # [n_pos_i, 1]
            dist_an_i = dist[i][diffenc[i]].unsqueeze(0) # [1, n_neg_i]
            diff_dist_i = dist_ap_i - dist_an_i # [n_pos_i, n_neg_i]
            loss_i = F.relu(diff_dist_i + self.margin) # [n_pos_i, n_neg_i]
            loss_avg_i = self.AvgNonZeroReducer(loss_i) # [1]
            total_loss += loss_avg_i

        return total_loss / batch_size
    
    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum()
        loss_num = (loss != 0).sum().float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg
        

    def forward(self, inputs: torch.Tensor,
                targets: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape
                (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape
                (num_classes).

        Returns:
            torch.Tensor: triplet loss.
        """
        if self.hard_mining:
            loss = self.hard_mining_triplet_loss_forward(inputs, targets)
        else:
            loss = self.no_hard_mining_triplet_loss_forward(inputs, targets)
        return loss * self.loss_term_weight
