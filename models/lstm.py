import sys
sys.path.append('./')
import torch
import torch.nn as nn
from models.base_model import BaseModel

class LSTM(BaseModel):
    def __init__(self, in_ch, hidden_dim, num_layers, num_classes, norm_linear):
        super(LSTM, self).__init__(hidden_dim, num_classes, norm_linear)
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        outs, _ = self.lstm(x)
        feats = outs[:, -1, :]  
        return self.linear(feats), feats # logits, feats
    
if __name__ == '__main__':
    lstm = LSTM(in_ch=13, hidden_dim=128, num_layers=3, num_classes=6, norm_linear=False)
    print(lstm)
    input = torch.randn(100, 64, 13)
    output = lstm(input)
    print(output[0].shape)