import losses
import torch.nn as nn
from utils.common import Odict, get_valid_args, is_dict, is_list


class FeatureLoss(nn.Module):

    def __init__(self, loss_term_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_term_weight = loss_term_weight

    def forward(self, features, labels):
        raise NotImplementedError("Subclasses should implement this method")
    
class ClassLoss(nn.Module):
    def __init__(self, loss_term_weight=1.0):
        super(ClassLoss, self).__init__()
        self.loss_term_weight = loss_term_weight

    def forward(self, logits, targets):
        raise NotImplementedError("Subclasses should implement this method")
    
    
class TrainingLoss(nn.Module):
    def __init__(self, loss_cfg):
        super(TrainingLoss, self).__init__()
        self.losses = nn.ModuleDict({loss_cfg['type']: self._build_loss_(loss_cfg)} if is_dict(loss_cfg) \
            else {cfg['type']: self._build_loss_(cfg) for cfg in loss_cfg})

    def _build_loss_(self, loss_cfg):
        """Build the losses from loss_cfg.

        Args:
            loss_cfg: Config of loss.
        """
        Loss = getattr(losses, loss_cfg['type'])
        valid_loss_arg = get_valid_args(Loss, loss_cfg, ['type'])
        loss = Loss(**valid_loss_arg)
        return loss
    
    def forward(self, logits, feats, targets):
        total_loss = 0
        loss_info = Odict()
        for loss_name, loss in self.losses.items():
            if isinstance(loss, FeatureLoss):
                loss_val = loss(feats, targets)
            elif isinstance(loss, ClassLoss):
                if is_list(logits):
                    loss_val = sum(loss(logit, targets) for logit in logits)
                else:
                    loss_val = loss(logits, targets)
            else:
                raise ValueError(f"Invalid loss type: {type(loss)}")
            total_loss += loss_val
            loss_info[loss_name] = loss_val
        return total_loss, loss_info