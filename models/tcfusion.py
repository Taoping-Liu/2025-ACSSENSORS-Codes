import torch
import torch.nn as nn
import sys
sys.path.append('./')
import models as models
from models.base_model import BaseModel
from utils.common import get_valid_args

class ECA(nn.Module):
    """
        codes from: https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
    """
    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 

    def forward(self, x):
        """
            x: shape(B, C) or shape(B, C, N)
        """
        if len(x.shape) == 2:
            y = x.unsqueeze(-1) # [B, C] -> [B, C, 1]
        else:
            y = torch.mean(x, dim=2, keepdim=True) # [B, C, N] -> [B, C, 1]

        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2) # [B, C, 1] -> [B, 1, C] -> [B, C, 1]

        y = torch.sigmoid(y) # [B, C, 1]
        
        if len(x.shape) == 2:
            y = y.squeeze(-1) # [B, C, 1] -> [B, C]

        return x * y # [B, C, N] * [B, C, 1] -> [B, C, N]

class Projector(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, act= None):
        super(Projector, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch) if bn else nn.Identity()
        self.act = nn.Identity() if act is None else act
        
    def forward(self, x):
        return self.act(self.bn(self.fc(x)))


class HybridFusion(nn.Module):
    def __init__(self, cnn_dim, transformer_dim, out_dim):
        super(HybridFusion, self).__init__()
        self.fc1 = Projector(cnn_dim, out_dim)
        self.fc2 = Projector(transformer_dim, out_dim)
        self.fc = Projector(out_dim*3, out_dim, bn=True, act=nn.ReLU())
        self.eca = ECA()

        
    def forward(self, cnn_x, transformer_x):
        cnn_x = self.fc1(cnn_x)
        transformer_x = self.fc2(transformer_x)
        
        fusion_add = cnn_x + transformer_x
        fuison_cat = torch.cat([cnn_x, transformer_x], dim=1) # [B, out_dim * 2]
        
        fusion_hybrid = torch.cat([fusion_add, fuison_cat], dim=1) # [B, out_dim * 3]
        out_eca = self.eca(fusion_hybrid) # [B, out_dim*3]
        return self.fc(out_eca) 
        
class AddFusion(nn.Module):
    def __init__(self, cnn_dim, transformer_dim, out_dim):
        super(AddFusion, self).__init__()
        self.fc1 = Projector(cnn_dim, out_dim)
        self.fc2 = Projector(transformer_dim, out_dim)
        self.fc = Projector(out_dim, out_dim, bn=True, act=nn.ReLU())
        
    def forward(self, cnn_x, transformer_x):
        return self.fc(self.fc1(cnn_x) + self.fc2(transformer_x))


class CatFusion(nn.Module):
    def __init__(self, cnn_dim, transformer_dim, out_dim):
        super(CatFusion, self).__init__()
        self.fc1 = Projector(cnn_dim, out_dim)
        self.fc2 = Projector(transformer_dim, out_dim)
        self.fc = Projector(out_dim*2, out_dim, bn=True, act=nn.ReLU())
        
    def forward(self, cnn_x, transformer_x):
        cnn_x = self.fc1(cnn_x)
        transformer_x = self.fc2(transformer_x)
        return self.fc(torch.cat([cnn_x, transformer_x], dim=1))

class TCFusion(BaseModel):
    def __init__(self, cnn_config, transformer_config, fusion='Cat', feat_dim = 128, num_classes=6, norm_linear=False, aux_loss=False):
        super(TCFusion, self).__init__(feat_dim, num_classes, norm_linear)
        
        CNN_Model = getattr(models, cnn_config['type'])
        valid_cnn_args = get_valid_args(CNN_Model, cnn_config, ['type', 'model_name'])
        self.cnn = CNN_Model(**valid_cnn_args)
        
        Transformer_Model = getattr(models, transformer_config['type'])
        valid_transformer_args = get_valid_args(Transformer_Model, transformer_config, ['type', 'model_name'])
        self.transformer = Transformer_Model(**valid_transformer_args)
        
        Fusion_Model = globals()[fusion + 'Fusion']
        self.fusion = Fusion_Model(self.cnn.feat_dim, self.transformer.feat_dim, feat_dim)
        
        self.aux_loss = aux_loss # 辅助分类损失
        
    def forward(self, x):
        cnn_logits, cnn_feat = self.cnn(x)
        transformer_logits, transformer_feat = self.transformer(x)
        
        feats = self.fusion(cnn_feat, transformer_feat)
        if self.aux_loss:
            return [self.linear(feats), cnn_logits, transformer_logits], feats
        return self.linear(feats), feats
            
    
if __name__ == '__main__':
    cnn_config = {
        'type': 'ResNet',
        'in_ch': 6,
        'blocks': [3, 4, 6, 3],
        'feat_dim': 512,
        'num_classes': 3,
        'norm_linear': True,
        'model_name': 'resnet34'
    }
    
    transformer_config = {
        'type': 'MyTransformer',
        'in_ch': 6,
        'num_layers': 4,
        'seq_len': 65,
        'd_model': 512,
        'nhead': 8,
        'dim_ffn': 2048,
        'dropout': 0.5,
        'feat_dim': 512,
        'num_classes': 3,
        'norm_linear': True,
        'model_name': 'MyTransformer'
    }
    
    model = TCFusion(cnn_config, transformer_config, fusion='Hybrid', feat_dim=128, num_classes=6, norm_linear=False).cuda()
    x = torch.randn(64, 65, 6).cuda() # (batch_size, seq_len, in_ch)
    y = model(x)[0]
    print(y.shape)