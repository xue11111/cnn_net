import torch
import torch.nn as nn


class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.module = nn.Sequential(
            self.nin_block(1, 96, 11, 4, 0),
            nn.MaxPool2d(3,stride=2),
            self.nin_block(96,256,5,1,2),
            nn.MaxPool2d(3,stride=2),
            self.nin_block(256,384,3,1,1),
            nn.MaxPool2d(3,stride=2),
            nn.Dropout(0.5),
            self.nin_block(384,10,3,1,1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )



    def forward(self,x):
        return self.module(x)



    def nin_block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=
        padding),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),
            nn.ReLU()
        )

model = NiN()
for layer in model.modules():
    print(layer)