# coding: utf-8

'''
    Project: Wavelet DNN
    Authors: Axel Masquelin
    Description:
'''

# libraries and dependencies
# --------------------------------------------
from sklearn.metrics import roc_curve, auc, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.tensor
import torchvision
import torch

import matplotlib.pyplot as plt

import seaborn as sns
import preprocessing
import numpy as np
import cv2, sys, utils, os, glob, pywt, math
# --------------------------------------------

def dwcfilter (numparr, m, wave):
    # print("Level: " + str(m))

    coeffs = pywt.dwt2(numparr, wave)
    LL, (LH, HL, HH) = coeffs

    if (m + 1 > 1): 
        for i in range(m):
            coeffs = pywt.dwt2(LL, wave)
            LL, (LH, HL, HH) = coeffs

    if (m < levels - 1):
        plot_3dmesh(LL, norm = "Wavelet LL " + str(m))
        plot_3dmesh(HL, norm = "Wavelet HL " + str(m))
        plot_3dmesh(LH, norm = "Wavelet LH " + str(m))
        plot_3dmesh(HH, norm = "Wavelet HH " + str(m))
    else:
        print(m)
        plot_3dscatter(LL, norm = "Wavelet LL " + str(m))
        plot_3dscatter(HL, norm = "Wavelet HL " + str(m))
        plot_3dscatter(LH, norm = "Wavelet LH " + str(m))
        plot_3dscatter(HH, norm = "Wavelet HH " + str(m))
        

    
def normalizePlanes(nparray):
    
    data = np.zeros((1, 64, 64))

    maxHU = np.max(nparray) 
    minHU = np.min(nparray)
    
    norm = (nparray - minHU) / (maxHU - minHU)
    norm[norm>1] = 1
    norm[norm<0] = 0
    
    data[0][:][:] = norm
    
    return data

def plot_3dmesh(nparray, norm = "Normalized"):
    '''
    '''
    # print(norm)
    for l in range(nparray.shape[0]):
        x = np.arange(0,nparray[l].shape[0],1)
        y = np.arange(0,nparray[l].shape[1],1)
        z = np.zeros((nparray[l].shape[0]*nparray[l].shape[1]))    
    
        count = 0
    
        for i in range(nparray.shape[1]):
            for j in range(nparray.shape[2]):
                z[count] = nparray[l][i][j]
                count += 1

        fig = plt.figure()
        ax = Axes3D(fig)
        X,Y = np.meshgrid(x,y)
        
        Z = z.reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis,
                               linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_title('Nodule surface - ' + str(norm))

def plot_3dscatter(nparray, norm = "Normalized"):
    '''
    '''
    # print(norm)
    for l in range(nparray.shape[0]):
        x = np.arange(0,nparray[l].shape[0],1)
        y = np.arange(0,nparray[l].shape[1],1)
        z = np.zeros((nparray[l].shape[0]*nparray[l].shape[1]))    
    
        count = 0
    
        for i in range(nparray.shape[1]):
            for j in range(nparray.shape[2]):
                z[count] = nparray[l][i][j]
                count += 1

        fig = plt.figure()
        ax = Axes3D(fig)
        X,Y = np.meshgrid(x,y)
        
        Z = z.reshape(X.shape)
        surf = ax.scatter(X, Y, Z)

        ax.set_title('Nodule surface - ' + str(norm))

if __name__ == '__main__':  
    imagespath = r'/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/*/*/*.jpg'
    resultpath = os.path.split(os.getcwd())[0] + r'/results/Nodule Surface/'
    img_list = glob.glob(imagespath)
    numparr = np.zeros((1,64,64))

    for n in range(len(img_list)):
        im = cv2.imread(img_list[n], 0)
        numparr[0][:][:] = im
        plot_3dmesh(numparr, norm = 'Raw')
        plot_3dmesh(normalizePlanes(numparr), norm = 'Normalized')
        
        levels = pywt.dwt_max_level(data_len = 64, filter_len = 'db1')
        print(levels)
        for m in range(levels):
            dwcfilter(normalizePlanes(numparr), m, wave = 'db1')
        
        cv2.imshow('images', im)
        plt.show()

