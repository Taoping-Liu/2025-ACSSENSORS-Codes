import torch
import torch.nn as nn
from models.base_model import BaseModel

class MLP(BaseModel):
    def __init__(self, in_ch, hidden_dim_list, activation, num_classes, norm_linear):
        super(MLP, self).__init__(hidden_dim_list[-1], num_classes, norm_linear)
        self.first_layer = nn.Linear(in_ch, hidden_dim_list[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim_list[i], hidden_dim_list[i+1]) for i in range(len(hidden_dim_list)-1)])
        self.activation = getattr(nn, activation)(inplace=True) if type(activation) == str else activation

    def forward(self, x):
        x = self.first_layer(x)
        x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x) 
        x = torch.mean(x, dim=1)
        return self.linear(x), x # logits, feats