'''
Project: Lung CT Wavelet Decomposition for Automated Nodule Categorization
Author: axemasquelin
Date: 10/25/2018

Function Definition: 
    
'''
####################################
import torchvision.models as models
import torch.nn as nn
import torchvision
import utils
import numpy as np
import cv2
####################################

class NoConv_4D(nn.Module):
    def __init__(self):
        
        super(NoConv_4D, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4*128*128, 500), #4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, 100), #4096,1000
            nn.ReLU(inplace=True),
            nn.Linear(100, 2), #1000,2
        )

    def forward(self, x):
        x = x.view(-1, 4*128*128)
        x = self.classifier(x)
        return x

class Conv_3Fil(nn.Module):
    def __init__(self):

        super(Conv_3Fil, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(192*256, 100), #4*25*25
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        x = self.features(x)
        #Insert Image View
        fil = x.cpu().detach().numpy()
        x = x.view(-1, 192*256) #4*25*25
        x = self.classifier(x)
        return x, fil

class Conv_98Fil(nn.Module):
    def __init__(self):

        super(Conv_98Fil, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 98, 3, padding = 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, stride=4),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(524*256, 500), #4*25*25
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((128, 128))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        #Insert Image View
        # fil = x.cpu().detach().numpy()
        # cv2.waitKey(1)
        # x = x.view(-1, 524*256) #4*25*25
        x = self.classifier(x)
        return x, #fil

class NoConv_256(nn.Module):
    def __init__(self):
        
        super(NoConv_256, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1*256*256, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        x = x.view(-1, 1*256*256)
        #x = x.log() #Usually found after classifier to try to stabilize unstable systems
        x = self.classifier(x)
        return x

class alexnet_conv1(nn.Module):

    def __init__(self):
        super(alexnet_conv1, self).__init__()
        net = torchvision.models.alexnet(pretrained = True)

        self.features = nn.Sequential(*[net.features[i] for i in range(3)])
        self.avgpool = nn.AdaptiveAvgPool2d((128, 128))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*128*128, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 64*128*128)
        x = self.classifier(x)
        return x

class alexnet_conv2(nn.Module):

    def __init__(self):
        super(alexnet_conv2, self).__init__()
        net = torchvision.models.alexnet(pretrained = True)

        self.features = nn.Sequential(*[net.features[i] for i in range(6)])
        self.avgpool = nn.AdaptiveAvgPool2d((128, 128))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(192*128*128, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 192*128*128)
        x = self.classifier(x)
        return x

class alexnet_conv3(nn.Module):

    def __init__(self):
        super(alexnet_conv3, self).__init__()
        net = torchvision.models.alexnet(pretrained = True)

        self.features = nn.Sequential(*[net.features[i] for i in range(8)])
        self.avgpool = nn.AdaptiveAvgPool2d((64, 64))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(384*64*64, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 384*64*64)
        x = self.classifier(x)
        return x

class alexnet_conv4(nn.Module):

    def __init__(self):
        super(alexnet_conv4, self).__init__()
        net = torchvision.models.alexnet(pretrained = True)

        self.features = nn.Sequential(*[net.features[i] for i in range(10)])
        self.avgpool = nn.AdaptiveAvgPool2d((64, 64))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*64*64, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256*64*64)
        x = self.classifier(x)
        return x

class alexnet(nn.Module):

    def __init__(self):
        super(alexnet, self).__init__()
        net = torchvision.models.alexnet(pretrained = True)

        self.features = net.features
        self.avgpool = nn.AdaptiveAvgPool2d((128, 128))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(192*128*128, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 2),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 192*128*128)
        x = self.classifier(x)
        return x
