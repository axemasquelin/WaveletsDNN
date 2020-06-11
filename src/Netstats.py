# coding: utf-8
""" MIT License """
'''
    Project: Wavelet DNN
    Authors: Axel Masquelin
    Description:
'''
# Libraries
# ---------------------------------------------------------------------------- #
from sklearn.metrics import roc_curve, auc, confusion_matrix
from training_testing import *
from architectures import *

import preprocessing
import utils

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.tensor
import torchvision
import torch

import matplotlib.pyplot as plt
import numpy as np
import cv2, sys, os
# ---------------------------------------------------------------------------- #

def mutlitest_stats():


if __name__ == '__main__':
    """
    Definition:
    Inputs:
    Outputs:
    """

    # Network Parameters
    models = [
            'Wave1',                # Single Level Wavelet Decomposition Layer extracting 4 features
            'Conv1',                # Convolutional Layer 4 Feature Extracted
            # 'inception_wave',       # Multi Level Wavelet Decomposition
            # 'inception_conv',       # Multiscale Convolutional Module.
            # 'AlexNet',              # Standard Alexnet Architecture with modified classifier
            # 'WalexNet',             # Wavelet Alexnet Architecture
            ]

    metrics = [                     # Select Metric to Evaluate
            'aucs',                 # Area under the Curve
            # 'Sensitivity',         # Network Senstivity
            # 'Specificity',         # Network Specificity
            # 'epoch            
            ]
    
    # Variable Flags
    create_violin = True
    check_stats = True

    # Dataframe Inits
    df = pd.DataFrame()              # General Dataframe to generate Bar-graph data
    df_alex = pd.DataFrame()         # AlexeNet Dataframe for violin plots
    df_walex = pd.DataFrame()        # WalexNet Dataframe for violin plots
    df_conv1 = pd.DataFrame()        # Conv Layer Dataframe for violin plots
    df_wave1 = pd.DataFrame()        # Wavelet Layer Dataframe for violin plots

    for root, dirs, files in os.walk(os.getcwd()[0] + "/results/", topdown = True):
        for name in files:
            if (name.endswith("aucs.csv")):
                header = name.split('_')[0]
                print(name.split('_')[0])
                headers.append(header)
                mean_ = []
                filename = os.path.join(root,name)
                with open(filename, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        for l in range(len(row)-1):
                            mean_.append(float(row[l+1]))
                df[header] = np.transpose(mean_)
                if (name.split('_')[0] == 'AlexNet'):
                    df_alex[name.split('_')[0]] = np.transpose(mean_)
                if (name.split('_')[0] == 'WalexNet'):
                    df_walex[name.split('_')[0]] = np.transpose(mean_)
                if (name.split('_')[0] == 'Conv1'):
                    df_conv1[name.split('_')[0]] = np.transpose(mean_)
                if (name.split('_')[0] == 'Wave1'):
                    df_wave1[name.split('_')[0]] = np.transpose(mean_)
                        