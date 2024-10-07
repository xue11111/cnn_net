import torch
import torchvision
import torch.nn as nn



pretrained_net = torchvision.models.resnet18(pretrained=True)

net = nn.Sequential(*list(pretrained_net.children()))[:-2]

x = torch.rand(1,3,320,480)

num_classes = 21

net.add_module('final_conv',nn.Conv2d(512,num_classes,kernel_size=1))

net.add_module('transpose_conv',
               nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,padding=16,stride=32))
