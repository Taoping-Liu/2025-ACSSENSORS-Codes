import sys
sys.path.append('./')
import torch
import torch.nn as nn
from models.base_model import BaseModel
from models.common import ResidualBlock
from utils.common import _weights_init

class ResNet(BaseModel):
    def __init__(self, in_ch, blocks, feat_dim, num_classes=6, norm_linear=False):
        super(ResNet, self).__init__(feat_dim, num_classes, norm_linear)

        # Initial convolutional layer with 7x kernels, 64 output channels, stride 2
        self.pre = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3, bias=False), # here we use 13 channels as input, the original use 6 channels
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1))

        # Constructing the four main layers of the ResNet
        self.layer1 = self._make_layer(64, 64, blocks[0])
        self.layer2 = self._make_layer(64, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, blocks[3], stride=2)
        
        # average pooling
        self.average_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.apply(_weights_init)

    def _make_layer(self, in_ch, out_ch, block_num, stride=1):
        """
        Helper function to create a layer of residual blocks.

        Parameters:
            in_ch (int): Number of input channels for the layer.
            out_ch (int): Number of output channels for the layer.
            block_num (int): Number of residual blocks in the layer.
            stride (int): Stride for the convolution in the first block of the layer.
        """
        
        # Shortcut connection to match the dimensions if needed
        shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm1d(out_ch),
            # nn.ReLU()
        ) if stride != 1 or in_ch != out_ch else None

        # Creating the residual blocks
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Permuting the dimensions to match the expected input shape
        x = x.permute(0, 2, 1) # [B, N, C] -> [B, C, N]
        
        # Passing the input through the layers
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Adaptive average pooling and flattening for the fully connected layer
        x = self.average_pooling(x)
        x = x.view(x.size(0), -1) # [B, C, 1] -> [B, C]
        return self.linear(x), x # logits, feats
 
if __name__ == "__main__":
    renset34_1 = ResNet(13, [3, 4, 6, 3], 512)
    print(renset34_1)
    from torchvision.models import resnet34 as resnet34_torch
    renset34_2 = resnet34_torch()
    print(renset34_2)
    input1 = torch.randn(1, 64, 13)
    output1 = renset34_1(input1)
