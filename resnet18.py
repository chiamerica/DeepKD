import torch 
from torch import nn 
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Type, Union
from collections import OrderedDict 


class SE(nn.Module):
    def __init__(self, in_channels, rd_ratio=4, rd_dim=None):
        super().__init__()
        rd_dim = rd_dim or in_channels // rd_ratio
        self.se = nn.Sequential(OrderedDict([
            ('gap', nn.AdaptiveAvgPool2d(1)),
            ('fc1', nn.Conv2d(in_channels, rd_dim, 1, bias=False)),
            ('bn', nn.BatchNorm2d(rd_dim)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc2', nn.Conv2d(rd_dim, in_channels, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

    def forward(self, x):
        return x * self.se(x)

class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.residual = nn.Sequential(OrderedDict[str, nn.Module]([
            ('conv1', nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(planes)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(planes, planes, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(planes))
        ]))
        self.shortcut = nn.Sequential(OrderedDict[str, nn.Module]([
            ('conv', nn.Conv2d(inplanes, planes, 1, stride, 0, bias=False)),
            ('bn', nn.BatchNorm2d(planes))
        ])) if stride != 1 or inplanes != planes else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.residual(x) + self.shortcut(x))
        

class ResNet18(nn.Module):

    def __init__(self, zero_init_residual: bool = False):
        super().__init__()
        self.stem = nn.Sequential(OrderedDict[str, nn.Module]([
            ('conv', nn.Conv2d(3, 64, 7, 2, 3, bias=False)),
            ('bn', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU(inplace=True)),
            ('maxpool', nn.MaxPool2d(3, 2, 1))
        ]))
        self.layer1 = nn.Sequential(OrderedDict[str, nn.Module]([
            ('block1', BasicBlock(64, 64, 1)),
            ('block2', BasicBlock(64, 64, 1))
        ]))
        self.layer2 = nn.Sequential(OrderedDict[str, nn.Module]([
            ('block1', BasicBlock(64, 128, 2)),
            ('block2', BasicBlock(128, 128, 1))
        ]))
        self.layer3 = nn.Sequential(OrderedDict[str, nn.Module]([
            ('block1', BasicBlock(128, 256, 2)),
            ('block2', BasicBlock(256, 256, 1))
        ]))
        self.layer4 = nn.Sequential(OrderedDict[str, nn.Module]([
            ('block1', BasicBlock(256, 512, 2)),
            ('block2', BasicBlock(512, 512, 1))
        ]))
        # self.classifier = nn.Linear(512, 1000)
        self.classifier = nn.Sequential(OrderedDict[str, nn.Module]([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(512, 1000))
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.residual.bn2.weight is not None:
                    nn.init.constant_(m.residual.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.classifier(x.mean(dim=(2, 3)))
        x = self.classifier(x)
        return x
