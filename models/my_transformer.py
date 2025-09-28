import sys
sys.path.append('./')
import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.common import LearnablePositionalEncoding, ConvEmbedding, SelfAttention, FeedForward

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class MyTransformer(BaseModel):
    def __init__(self, in_ch = 13, num_layers= 3,  seq_len = 64, d_model = 128, nhead = 8, dim_ffn = 512, dropout = 0.1, feat_dim = 128,  num_classes=6, norm_linear=False):
        super(MyTransformer, self).__init__(feat_dim, num_classes, norm_linear)
        # add a class token, just like BERT or ViT
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.embedding = ConvEmbedding(in_ch, d_model)
        self.positional_encoding = LearnablePositionalEncoding(d_model, seq_len+1)
        self.transformer = TransformerEncoder(d_model, num_layers, nhead, d_model // nhead, dim_ffn, dropout)
        self.dropout = nn.Dropout(dropout)  

    def forward(self, x):
        x = self.embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        x = self.transformer(x)
        x = x[:, 0] # cls token
        
        return self.linear(x), x 

if __name__ == '__main__':
    model = MyTransformer()
    print(model)
    input = torch.randn(100, 64, 13)
    output = model(input)
    print(output[0].shape)