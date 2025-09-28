import torch.nn as nn
from models.common import NormedLinear

class BaseModel(nn.Module):
    def __init__(self, feat_dim, num_classes=6, norm_linear=False):
        super(BaseModel, self).__init__()
        self.linear = NormedLinear(feat_dim, num_classes) if norm_linear else nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
    
    def forward(self, x):
        return x