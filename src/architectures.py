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
        
        x = utils.tensor_cat(x1,x2, padding = False)

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

        x = utils.tensor_cat(x1,x2,x3, padding = False)
        
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

        x = utils.tensor_cat(x1,x2,x3,x4, padding = False)
        
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

        x = utils.tensor_cat(x1,x2,x3,x4,x5, padding = False)
        
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

        x = utils.tensor_cat(x1,x2,x3,x4,x5,x6, padding = False)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class Conv_1 (nn.Module):
    def __init__(self):
        
        super(Conv_1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 2, stride= 2, padding = 1),
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
            nn.Conv2d(1, 4, kernel_size= 2, stride= 2, padding = 0, dilation = 1),
            # nn.Conv2d(1, 4, kernel_size= 3, stride= 2, padding = 1, dilation = 1),
            nn.ReLU(inplace = True),
            # nn.MaxPool2d(kernel_size = 3, stride = 2),        
        )
        self.block3x3 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 4, stride= 4, padding = 2, dilation = 2),
            # nn.Conv2d(1, 4, kernel_size= 5, stride= 2, padding = 2, dilation = 1),
            nn.ReLU(inplace = True),
            # nn.MaxPool2d(kernel_size = 4, stride = 2), 
        )
        self.block5x5 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 8, stride= 8, padding = 8, dilation = 3),
            # nn.Conv2d(1, 4, kernel_size= 7, stride= 2, padding = 3, dilation = 1),
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

        x = utils.tensor_cat(x2,x3,x5, padding = False)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class incept_dilationconv(nn.Module):
    def __init__(self):
        
        super(incept_dilationconv, self).__init__()
        self.block2x2 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 2, stride= 2, padding = 0, dilation = 1),
            nn.ReLU(inplace = True),
            # nn.MaxPool2d(kernel_size = 3, stride = 2),        
        )
        self.block3x3 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 2, stride= 4, padding = 1, dilation = 2),
            nn.ReLU(inplace = True),
            # nn.MaxPool2d(kernel_size = 4, stride = 2), 
        )
        self.block5x5 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size= 2, stride= 8, padding = 2, dilation = 4),
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

        x = utils.tensor_cat(x2,x3,x5, padding = False)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
