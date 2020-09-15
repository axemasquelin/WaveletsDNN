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
# from statsmodels import stats
from training_testing import *
from architectures import *

import preprocessing
import utils

import statsmodels.stats.multitest as smt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import cv2, sys, os, csv
# ---------------------------------------------------------------------------- #

def multitest_stats(data1, data2):
    """
    Definition:
    Inputs:
    Outputs:
    """

    pvals = np.zeros(len(data1))
    tvals = np.zeros(len(data1))

    for i in range(len(data1)):
        t, p = scipy.stats.ttest_ind(data1[i, 1:], data2[i, 1:])
        pvals[i] = p
        tvals[i] = t
    
    y = smt.multipletests(p, alpha=0.01, method='b', is_sorted = False, returnsorted = False)
    print(y)
    # print("T-value: ", t)
    # print("P-value: ", p)

    return y



def violin_plots(df, metric, methods, sig_sl = None, sig_ml = None):
    """
    Definitions:
    Inputs:
    Outputs:
    """
    plt.figure()

    cols = [df.columns[-1]] + [col for col in df if col != df.columns[-1]]
    df = df[cols]

    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    my_pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}

    sns.violinplot(data = df, inner="quartile", fontsize = 15, palette= sns.color_palette("RdBu_r", 7)) #bw = 0.15

    plt.title(metric + " Distribution Across Methodologies")
    plt.xlabel("Methodology", fontsize = 12)
    plt.ylabel(metric, fontsize = 12)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if sig_sl != None:
        if sig_sl[1][0] > 0.05:
            sigtext = 'ns'
        elif (sig_sl[1][0] < 0.05 and sig_sl[1][0] > 0.01):
            sigtext = '*'
        elif (sig_sl[1][0] < 0.01 and sig_sl[1][0] > 0.001): 
            sigtext = '**'
        elif sig_sl[1][0] < 0.001: 
            sigtext = '***'
        x1, x2 = 0, 1
        y, h, col = .945, .005, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, sigtext , ha='center', va='bottom', color=col)

    if sig_ml != None:
        if sig_ml[1][0] > 0.05:
            sigtext = 'ns'
        elif (sig_ml[1][0] < 0.05 and sig_ml[1][0] > 0.001):
            sigtext = '*'
        elif (sig_ml[1][0] < 0.01 and sig_ml[1][0] > 0.001): 
            sigtext = '**'
        elif sig_ml[1][0] < 0.001: 
            sigtext = '***'
        x1, x2 = 2, 3
        y, h, col = .945, .005, 'k'
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, sigtext, ha='center', va='bottom', color=col)
    
    plt.savefig(os.path.split(os.getcwd())[0] + "/results/" + metric + "_Across_Methods.png")

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
    np_conv1 = np.zeros((5,5))        # Conv Layer Dataframe for violin plots
    np_wave1 = np.zeros((5,5))        # Wavelet Layer Dataframe for violin plots
    np_wavecept = np.zeros((5,5))     # Multi-level Wavelet Dataframe
    np_convcept = np.zeros((5,5))     # Multi-level Convolutional Dataframe

    print(os.path.split(os.getcwd()))

    for root, dirs, files in os.walk(os.path.split(os.getcwd())[0] + "/results/", topdown = True):
        for name in files:
            if (name.endswith(metrics[0] + ".csv")):
                header = name.split('_aucs')[0]
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
                    if (header == 'Conv1'):
                        np_conv1 = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                    if (header == 'Wave1'):
                        np_wave1 = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                    if (header == 'inception_conv'):
                        np_convcept = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                    if (header == 'inception_wave'):
                        np_wavecept =  np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
    print(np_wavecept)
                    
    print(df)
    if check_stats:
        print("Comparing Single-level Analysis")
        ssl = multitest_stats(np_wave1, np_conv1)
        
        print("Comparing Multi-level Analysis")
        sml = multitest_stats(np_wavecept, np_convcept)
            
    if create_violin:
        print("Violin Plots")
        if check_stats:
            violin_plots(df, metrics[0], models, sig_sl = ssl, sig_ml = sml)
        else:
            violin_plots(df, metrics[0], models)