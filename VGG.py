import torch
import torch.nn as nn


conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()

        self.conv_blks = []
        self.in_channels = 1
        for (num_convs,out_channels) in conv_arch:
            self.conv_blks.append(self.vgg_block(num_convs,self.in_channels,out_channels))
            self.in_channels = out_channels

        self.module = nn.Sequential(
            *self.conv_blks,
            nn.Flatten(),
            nn.Linear(out_channels * 7 * 7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)

        )
        self._initialize_weights()


    def forward(self,x):
        return self.module(x)



    def vgg_block(self,num_convs,in_channels,out_channels):
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels,out_channels,kernel_size=3,padding=1
            ))
            layers.append(nn.ReLU())
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        module_i = nn.Sequential(*layers)
        return module_i

    # 定义权重初始化函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


model = Vgg()
for layer in model.modules():
    if isinstance(layer,(nn.Conv2d,nn.Linear)):
        print(layer)