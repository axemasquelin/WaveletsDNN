# coding: utf-8
""" MIT License """
'''
    Project: Wavelet DNN
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
import torchvision.models as models
import torch.nn as nn
import torchvision
import torch

import preprocessing as pre
import numpy as np
import utils
import cv2
# ---------------------------------------------------------------------------- #
class Wave_1(nn.Module):
    def __init__(self):
        
        super(Wave_1, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4*6*6, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 2),
        )

    def forward(self, x, device):
        x = pre.singlelvl_wd(x, device, int(x.size()[2]/2))
        # print(x.size())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class incept_wave2(nn.Module):
    def __init__(self):
        
        super(incept_wave2, self).__init__()
        self.nonlin = nn.ReLU(inplace = True)
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8*6*6, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 2),
        )

    def forward(self, x, device):
        LL, x1 = pre.multiscale_wd(x, device, int(x.size()[2]/2))
        LL, x2 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        # LL, x3 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))

        x = utils.tensor_cat(x1,x2)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class incept_wave3(nn.Module):
    def __init__(self):
        
        super(incept_wave3, self).__init__()
        self.nonlin = nn.ReLU(inplace = True)
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(12*6*6, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 2),
        )

    def forward(self, x, device):
        LL, x1 = pre.multiscale_wd(x, device, int(x.size()[2]/2))
        LL, x2 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x3 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))

        x = utils.tensor_cat(x1,x2,x3)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class incept_wave4(nn.Module):
    def __init__(self):
        
        super(incept_wave4, self).__init__()
        self.nonlin = nn.ReLU(inplace = True)
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16*6*6, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 2),
        )

    def forward(self, x, device):
        LL, x1 = pre.multiscale_wd(x, device, int(x.size()[2]/2))
        LL, x2 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x3 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x4 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))

        x = utils.tensor_cat(x1,x2,x3,x4)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class incept_wave5(nn.Module):
    def __init__(self):
        
        super(incept_wave5, self).__init__()
        self.nonlin = nn.ReLU(inplace = True)
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(20*6*6, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 2),
        )

    def forward(self, x, device):
        LL, x1 = pre.multiscale_wd(x, device, int(x.size()[2]/2))
        LL, x2 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x3 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x4 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x5 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))

        x = utils.tensor_cat(x1,x2,x3,x4,x5)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class incept_wave6(nn.Module):
    def __init__(self):
        
        super(incept_wave6, self).__init__()
        self.nonlin = nn.ReLU(inplace = True)
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(24*6*6, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 2),
        )

    def forward(self, x, device):
        LL, x1 = pre.multiscale_wd(x, device, int(x.size()[2]/2))
        LL, x2 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x3 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x4 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x5 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))
        LL, x6 = pre.multiscale_wd(LL, device, int(LL.size()[2]/2))

        x = utils.tensor_cat(x1,x2,x3,x4,x5,x6)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class Conv_1 (nn.Module):
    def __init__(self):
        
        super(Conv_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 3, stride= 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4*6*6, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 2),
        )

    def forward(self, x, device):
        # print(x.size())
        x = self.features(x)
        # print(x.size())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        

class incept_conv(nn.Module):
    def __init__(self):
        
        super(incept_conv, self).__init__()
        self.block2x2 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 2, stride= 2, padding = 0),
            nn.ReLU(inplace = True),
            # nn.MaxPool2d(kernel_size = 3, stride = 2),        
        )
        self.block3x3 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 4, stride= 4, padding = 0),
            nn.ReLU(inplace = True),
            # nn.MaxPool2d(kernel_size = 4, stride = 2), 
        )
        self.block5x5 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 8, stride= 8, padding = 0),
            nn.ReLU(inplace = True),
            # nn.MaxPool2d(kernel_size = 8, stride = 2), 
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(12*6*6, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(150, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 2),
        )

    def forward(self, x, device):
        x2 = self.block2x2(x)
        x3 = self.block3x3(x)
        x5 = self.block5x5(x)

        x = utils.tensor_cat(x2,x3,x5)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= 11, stride= 4, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(64, 192, kernel_size = 5, padding= 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride = 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, 250),
            nn.ReLU(inplace = True),
            nn.Linear(250, num_classes),
        )

    def forward(self,x, device):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

class WalexNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(WalexNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 61, kernel_size= 2, stride= 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 189, kernel_size = 2, padding= 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            )
        self.block4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 253, kernel_size = 2, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride = 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 250),
            nn.ReLU(inplace=True),
            nn.Dropout()
            ,nn.Linear(250, 50),
            nn.ReLU(inplace = True),
            nn.Linear(50, num_classes),
        )

    def forward(self,x, device):
        # Block 1 
        LL, x = pre.multiscale_wd(x, device, int(x.size()[2]/2))
        x = self.block1(x)
        
        # Block 2
        LL, x1 = pre.multiscale_wd(LL, device, x.size()[2])
        x = torch.cat((x, x1), 1)
        x = self.block2(x)
        
        # Block 3
        LL, x1 = pre.multiscale_wd(LL, device, x.size()[2])
        x = torch.cat((x, x1), 1)
        x = self.block3(x)
        
        # Block 4
        x = self.block4(x)
        LL, x1 = pre.multiscale_wd(LL, device, x.size()[2])
        x = torch.cat((x, x1), 1)

        # Average Pool and Flatten
        x = self.avgpool(x) 
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x