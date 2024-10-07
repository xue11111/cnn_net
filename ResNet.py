import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, input_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(input_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,out_channels,kernel_size=1,stride=stride)

        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)




    def forward(self,x):

        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y += x
        return self.relu(y)

blk = Residual(3,3)
x = torch.randn(4,3,6,6)
y = blk(x)
print(y.shape)