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

import matplotlib.pyplot as plt
import statsmodels as sm
import pandas as pd
import numpy as np
import scipy
import cv2, sys, os, csv
# ---------------------------------------------------------------------------- #

def multitest_stats(df1, df2):
    """
    Definition:
    Inputs:
    Outputs:
    """
    print(df1['aucs'])
    t, p = scipy.stats.ttest_ind(df1['aucs'], df2['aucs'])
    print(p)

def violin_plots():
    """
    Definitions:
    Inputs:
    Outputs:
    """


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
            'inception',
            'inception_wave',       # Multi Level Wavelet Decomposition
            'inception_conv',       # Multiscale Convolutional Module.
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

    # Dataframe Inits_
    df = pd.DataFrame()              # General Dataframe to generate Bar-graph data
    df_conv1 = pd.DataFrame()        # Conv Layer Dataframe for violin plots
    df_wave1 = pd.DataFrame()        # Wavelet Layer Dataframe for violin plots
    df_wavecept = pd.DataFrame()     # Multi-level Wavelet Dataframe
    df_convcept = pd.DataFrame()     # Multi-level Convolutional Dataframe

    print(os.path.split(os.getcwd()))

    for root, dirs, files in os.walk(os.path.split(os.getcwd())[0] + "/results/", topdown = True):
        for name in files:
            if (name.endswith(metrics[0] + ".csv")):
                header = name.split('_')[0]
                if header in models:
                    mean_ = []
                    filename = os.path.join(root,name)
                    with open(filename, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            for l in range(len(row)-1):
                                mean_.append(float(row[l+1]))
                    df[header] = np.transpose(mean_)
                    if (name.split('_')[0] == 'Conv1'):
                        df_conv1[metrics[0]] = np.transpose(mean_)
                    if (name.split('_')[0] == 'Wave1'):
                        df_wave1[metrics[0]] = np.transpose(mean_)
                    if (name.split('_')[0] == 'inception'):
                        if(name.split('_')[1] == 'conv'):
                            df_convcept[metrics[0]] = np.transpose(mean_)
                        if(name.split('_')[1] == 'wave'):
                            df_wavecept[metrics[0]] = np.transpose(mean_)

    if check_stats:
        print("Comparing Single-level Analysis")
        multitest_stats(df_wave1, df_conv1)
        
        print("Comparing Multi-level Analysis")
        multitest_stats(df_wavecept, df_convcept)
            
    if create_violin:
        print("Violin Plots")

                        
