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
import string
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

def annotatefig(sig, x1, x2, y, h):
    if sig < 0.05:
        if (sig < 0.05 and sig > 0.01):
            sigtext = '*'
        elif (sig < 0.01 and sig > 0.001): 
            sigtext = '**'
        elif sig < 0.001: 
            sigtext = '***'

        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        plt.text((x1+x2)*.5, y+h, sigtext , ha='center', va='bottom', color='k')

def violin_plots(df, metric, methods, sig_sl = None, sig_ml = None, sig_wl = None, sig_cl = None):
    """
    Definitions:
    Inputs:
    Outputs:
    """
    plt.figure()

    # cols = [df.columns[-1]] + [col for col in df if col != df.columns[-1]]
    # df = df[cols]

    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    my_pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}

    sns.violinplot(data = df, inner="quartile", fontsize = 16, palette= sns.color_palette("RdBu_r", 7)) #bw = 0.15
    plt.xlabel("Methodology", fontsize = 12)
    
    if metric == 'auc':
        plt.title(metric.upper() + " Distribution Across Methodologies")
        plt.ylabel(metric.upper(), fontsize = 12)
    else:
        plt.title(metric.capitalize() + " Distribution Across Methodologies")
        plt.ylabel(metric.capitalize(), fontsize = 12)    
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    if sig_sl != None:
        x1, x2 = 0, 2
        y, h, col = .885, .005, 'k'
        annotatefig(sig_sl[1][0], x1, x2, y, h)

    if sig_ml != None:
        x1, x2 = 1, 3
        y, h, col = .915, .005, 'k'
        annotatefig(sig_ml[1][0], x1, x2, y, h)

    if sig_wl != None:
        x1, x2 = 2, 3
        y, h, col = .905, .005, 'k'
        annotatefig(sig_wl[1][0], x1, x2, y, h)

    if sig_cl != None:
        x1, x2 = 0, 1
        y, h, col = .90, .005, 'k'
        annotatefig(sig_cl[1][0], x1, x2, y, h)
    
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
            'Wave2',                # Multi level Wavelet Decomposition Layer extracting 4 features
            'Wave3',                # Multi level Wavelet Decomposition Layer extracting 4 features
            'Wave4',                # Multi level Wavelet Decomposition Layer extracting 4 features
            'Wave5',                # Multi level Wavelet Decomposition Layer extracting 4 features
            'Wave6',                # Multi level Wavelet Decomposition Layer extracting 4 features
            'Conv1',                # Convolutional Layer 4 Feature Extracted
            'Conv3',
            ]

    metrics = [                    # Select Metric to Evaluate
            'auc',                 # Area under the Curve
            'sensitivity',         # Network Senstivity
            'specificity',         # Network Specificity         
            'time',
            ]
    
    # Variable Flags
    create_violin = True
    check_stats = True
    print(os.path.split(os.getcwd()))
    
    for metric in metrics:
        print(metric)
        # Dataframe Inits_
        df = pd.DataFrame()               # General Dataframe to generate Bar-graph data
        np_conv1 = np.zeros((5,5))        # Conv Layer Dataframe for violin plots
        np_wave1 = np.zeros((5,5))        # Wavelet Layer Dataframe for violin plots
        np_wavecept = np.zeros((5,5))     # Multi-level Wavelet Dataframe
        np_convcept = np.zeros((5,5))     # Multi-level Convolutional Dataframe

        for root, dirs, files in os.walk(os.path.split(os.getcwd())[0] + "/results/", topdown = True):
            for name in files:
                if (name.endswith(metric + ".csv")):
                    header = name.split('_' + metric)[0]
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
                        print(header)
                        if metric == 'auc':
                            if (header == 'Conv1'):
                                np_conv1 = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                            if (header == 'Wave1'):
                                np_wave1 = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                            if (header == 'Conv3'):
                                np_convcept = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
                            if (header == 'Wave3'):
                                np_wavecept =  np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=1)
        
        df2 = df
        df2 = df2.drop(['Wave2', 'Wave4','Wave5','Wave6'], axis = 1)        
        
        # print(df)
        if check_stats:
            print("Comparing Single-level Analysis")
            ssl = multitest_stats(np_wave1, np_conv1)
            
            print("Comparing Multi-level Analysis")
            sml = multitest_stats(np_wavecept, np_convcept)
            
            swl = multitest_stats(np_wavecept, np_wave1)
            scl = multitest_stats(np_conv1, np_convcept)
        if create_violin:
            print("Violin Plots")
            if (check_stats and metric == 'auc'):
                violin_plots(df2, metric, models, sig_sl = ssl, sig_ml = sml, sig_wl = swl, sig_cl = scl)
            else:
                violin_plots(df, metric, models)

        if metric == 'sensitivity':
            print("Wave1: ", df['Wave1'].mean())
            print("Wave2: ", df['Wave2'].mean())
            print("Wave3: ", df['Wave3'].mean())
            print("Wave4: ", df['Wave4'].mean())
            print("Wave5: ", df['Wave5'].mean())
            print("Wave6: ", df['Wave6'].mean())
        
        if metric == 'specificity':
            print("Wave1: ", df['Wave1'].mean())
            print("Wave2: ", df['Wave2'].mean())
            print("Wave3: ", df['Wave3'].mean())
            print("Wave4: ", df['Wave4'].mean())
            print("Wave5: ", df['Wave5'].mean())
            print("Wave6: ", df['Wave6'].mean())
        
        if metric == 'time':
            print("Wave1: ", df['Wave1'].mean())
            print("Conv1: ", df['Conv1'].mean())
            