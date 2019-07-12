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

import numpy as np
import cv2
####################################

class NoConv(nn.Module):
    def __init__(self, setting):
        
        super(NoConv, self).__init__()
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
