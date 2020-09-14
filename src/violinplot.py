# coding: utf-8
""" MIT License """
'''
    Project: Wavelet DNN
    Authors: Axel Masquelin
    Description:
'''

# libraries and dependencies
# ---------------------------------------------------------------------------- #
from scipy import stats
# from tables import *

import matplotlib.pyplot as plt
import matplotlib.axes as ax
# import seaborn as sns
import pandas as pd
import numpy as np
import glob, os, csv, re
# ---------------------------------------------------------------------------- #\
def violin(df, method):
    """
    Description:
    Inputs:
    Outputs:
    """
    plt.figure()
    
    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    my_pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}

    sns.violinplot(data = df, inner="quartile", fontsize = 15, palette= sns.color_palette("RdBu_r", 7)) #bw = 0.15

    plt.title(metric + " Across " + method)
    plt.xlabel("Dataset", fontsize = 12)
    plt.ylabel(metric, fontsize = 12)
    plt.savefig(os.getcwd() + "/results/" + metric + " Across " + method)

def gen_bars(data, headers, datasets, methods):
    """
    Description:
    Inputs:
    Outputs:
    """
    width = 0.5
    x = np.arange(0, 16, 2)
    
    labels_x = datasets.copy()
    labels_x.insert(0,"") 
    
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - (0.3 + width/2), data[0,:], width, label = methods[0])
    add_labels(ax, bar1)

    if (len(methods) > 1):
        bar2 = ax.bar(x , data[1,:], width, label = methods[1])
        add_labels(ax, bar2)
    
        if (len(methods) > 2):
            bar3 = ax.bar(x + (0.3 + width/2), data[1,:], width, label = methods[2])
            add_labels(ax, bar3)

    ax.set_ylabel(metric)
    ax.set_xlabel("Dataset")
    
    ax.set_xticklabels(labels_x)
    ax.set_title(metric + " Across Datasets")
    ax.legend(bbox_to_anchor=(1, 1.15), fancybox=True, framealpha=0.5)

def add_labels(ax, bars):
    """
    Description:
    Inputs:
    Outputs:
    """
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def dataset_analysis(df1, df2, method1, method2):
    df_stats = pd.DataFrame()
    
    gprs = list(df1.columns.values)
    for grp in gprs:
        names = {'A': [method1 + "-" + grp], 'B': [method2 + "-" + grp]}
        df_temp = pd.DataFrame(data = names)
        
        meanA = df1[grp].mean()
        meanB = df2[grp].mean()
        diff = meanA - meanB

        t, p = stats.f_oneway(df1[grp], df2[grp])

        df_temp['mean (A)'] = meanA
        df_temp['mean (B)'] = meanB
        df_temp['diff'] = diff
        df_temp['T'] = t
        df_temp['p'] = p
        df_stats = df_stats.append(df_temp, ignore_index= True)

    df_stats.to_csv(os.getcwd()[0] + "/results/" + method1 + '_' + method2 + ".csv")

def method_analysis(data, datasets, method):
    df_stats = pd.DataFrame()
    
    gprs = list(data.columns.values)
    for grp1 in gprs:
        for grp2 in gprs:
            if grp1 != grp2:
                names = {'A': [grp1], 'B': [grp2]}
                df_temp = pd.DataFrame(data = names)
             
                meanA = data[grp1].mean()
                meanB = data[grp2].mean()
                diff = meanA - meanB
                t, p = stats.f_oneway(data[grp1], data[grp2])

                df_temp['mean (A)'] = meanA
                df_temp['mean (B)'] = meanB
                df_temp['diff'] = diff
                df_temp['T'] = t
                df_temp['p'] = p
                df_stats = df_stats.append(df_temp, ignore_index= True)             
    df_stats.to_csv(os.getcwd()[0] + "/results/" + method + ".csv")


if __name__ == '__main__':

    metrics = [
               'aucs',              # Select Metric to Evaluate
            #    'Sensitivity',
            #    'Specificity'
              ]
    
    create_violin = True
    check_stats = True
    plot_auc = True

    df = pd.DataFrame()         # General Dataframe to generate Bar-graph data
    df_alex = pd.DataFrame()     # SVM Dataframe for violin plots
    df_walex = pd.DataFrame()      # RF Dataframe for violin plots

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
                        

            # for n in range(2):
            #     specs[m,n] = round(np.mean(df[headers[n]]), 2)
            # m += 1
        
        # gen_bars(specs, headers, datasets, methods)
    print(df)
    if create_violin:
        if check_stats:
            # violin(df_alex, method = 'AlexNet')
            # violin(df_walex, method = 'WalexNet')
            dataset_analysis(df_alex, df_walex, method1 = 'AlexNet', method2 = 'WalexNet')
        if plot_auc:


    # plt.show()



