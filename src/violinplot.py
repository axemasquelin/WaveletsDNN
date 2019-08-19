# coding: utf-8

""" MIT License """
'''
    Axel Masquelin & Sami Connolly  
    Andrea Elhajj  & Thayer Alshaabi
    ---
    Description:
    Generates violin plots of AUC data stored in csv files, and runs a two way t-test to
    evaluate whether two approaches are statistically different from each other. 
    ---
    ---
    Copyright (c) 2018 
'''

# libraries and dependencies
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import seaborn as sns
import pandas as pd
from scipy import stats
import numpy as np
import glob
import csv
import os
import re
# ---------------------------------------------------------------------------- #\
df = pd.DataFrame()
cwd = os.getcwd()
base = []
print(os.path.split(os.getcwd())[0] )
for root, dirs, files in os.walk(os.path.split(os.getcwd())[0]+ "/results/08-05-2019", topdown = True):
    plt.figure(1)
    for name in files:
        if (name.endswith(".csv")):
            if (len(name.split('_')) == 2):
                header = name.split('_')[0]
                print(header)
                mean_AUC = []
                filename = os.path.join(root,name)
                base.append(os.path.splitext(os.path.basename(filename))[0])
                #aucs = csv.read_csv(filename)
                with open(filename, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        for l in range(len(row)-1):
                            mean_AUC.append(float(row[l+1]))
                df[header] = np.transpose(mean_AUC)



print(np.mean(df['Conv3']))
print(np.mean(df['Alex1']))
print(np.mean(df['Alex2']))

# print("Effect Size: " + str(np.mean(df['aeOrigin']) - np.mean(df['aetop16'])))
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
my_pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}

# labels = {'ae', 'rf', 'svm'}
sns.violinplot(data = df, inner="quartile", fontsize = 20, palette= sns.color_palette("RdBu_r", 7)) #bw = 0.15
# plt.ylim((0,1))
plt.title('AUC Distribution of All Approaches')
plt.xlabel('Methods', fontsize = 12)
plt.ylabel('AUC', fontsize = 12)

# #T-Test
# #Alex1
# t,p = stats.ttest_ind(df['Alex1'], df['Alex2'], equal_var = False)
# print("Alex1, Alex2: ", t,p)
# t,p = stats.ttest_ind(df['Alex1'], df['Alex3'], equal_var = False)
# print("Alex1, Alex3: ", t,p)
# t,p = stats.ttest_ind(df['Alex1'], df['Alex4'], equal_var = False)
# print("Alex1, Alex4: ", t,p)
# t,p = stats.ttest_ind(df['Alex1'], df['Raw'], equal_var = False)
# print("Alex1, Raw: ", t,p)
# t,p = stats.ttest_ind(df['Alex1'], df['WaveF'], equal_var = False)
# print("Alex1, WaveF: ", t,p)
# t,p = stats.ttest_ind(df['Alex1'], df['Conv3'], equal_var = False)
# print("Alex1, Conv3: ", t,p)

# #Alex2
# t,p = stats.ttest_ind(df['Alex2'], df['Alex3'], equal_var = False)
# print("Alex2, Alex3: ", t,p)
# t,p = stats.ttest_ind(df['Alex2'], df['Alex4'], equal_var = False)
# print("Alex2, Alex4: ", t,p)
# t,p = stats.ttest_ind(df['Alex2'], df['Raw'], equal_var = False)
# print("Alex2, Raw: ", t,p)
# t,p = stats.ttest_ind(df['Alex2'], df['WaveF'], equal_var = False)
# print("Alex2, WaveF: ", t,p)
# t,p = stats.ttest_ind(df['Alex2'], df['Conv3'], equal_var = False)
# print("Alex2, Conv3: ", t,p)

# #Alex3
# t,p = stats.ttest_ind(df['Alex3'], df['Alex4'], equal_var = False)
# print("Alex3, Alex4: ", t,p)
# t,p = stats.ttest_ind(df['Alex3'], df['Raw'], equal_var = False)
# print("Alex3, Raw: ", t,p)
# t,p = stats.ttest_ind(df['Alex3'], df['WaveF'], equal_var = False)
# print("Alex3, WaveF: ", t,p)
# t,p = stats.ttest_ind(df['Alex3'], df['Conv3'], equal_var = False)
# print("Alex3, Conv3: ", t,p)

# #Alex4
# t,p = stats.ttest_ind(df['Alex4'], df['Raw'], equal_var = False)
# print("Alex4, Raw: ", t,p)
# t,p = stats.ttest_ind(df['Alex4'], df['WaveF'], equal_var = False)
# print("Alex4, WaveF: ", t,p)
# t,p = stats.ttest_ind(df['Alex4'], df['Conv3'], equal_var = False)
# print("Alex4, Conv3: ", t,p)

# #WaveF
# t,p = stats.ttest_ind(df['WaveF'], df['Raw'], equal_var = False)
# print("WaveF, Raw: ", t,p)
# t,p = stats.ttest_ind(df['WaveF'], df['Conv3'], equal_var = False)
# print("WaveF, Conv3: ", t,p)

# #Conv3
# t,p = stats.ttest_ind(df['Conv3'], df['Raw'], equal_var = False)
# print("Conv3, Raw: ", t,p)


plt.show()

