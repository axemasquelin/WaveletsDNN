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

import matplotlib.pyplot as plt

import seaborn as sns
import preprocessing
import numpy as np
import cv2, sys, utils, os, glob, pywt, math
# --------------------------------------------

def dwcfilter (numparr, m, savepath, wave):
    # print("Level: " + str(m))

    coeffs = pywt.dwt2(numparr, wave)
    LL, (LH, HL, HH) = coeffs
    cv2.imshow('LL', LL[0]*1.2)
    cv2.imshow('LH', LH[0]*1.2)    
    cv2.imshow('HL', HL[0]*1.2)
    cv2.imshow('HH', HH[0]*1.2)

    PILimg_LL = Image.fromarray(LL[0]*1.2)
    PILimg_LH = Image.fromarray(LH[0]*1.2)
    PILimg_HL = Image.fromarray(HL[0]*1.2)
    PILimg_HH = Image.fromarray(HH[0]*1.2)
    
    PILimg_LL = PILimg_LL.convert('RGB')
    PILimg_LH = PILimg_LH.convert('RGB')
    PILimg_HL = PILimg_HL.convert('RGB')
    PILimg_HH = PILimg_HH.convert('RGB')
    
    PILimg_LL.save('result_LL.png', dpi = (600,600))
    PILimg_LH.save('result_LH.png', dpi = (600,600))
    PILimg_HL.save('result_HL.png', dpi = (600,600))
    PILimg_HH.save('result_HH.png', dpi = (600,600))

    # plt.show()

    if (m + 1 > 1): 
        for i in range(m):
            coeffs = pywt.dwt2(LL, wave)
            LL, (LH, HL, HH) = coeffs

            

    if (m < levels - 1):
        plot_3dmesh(LL, m, savepath, norm = "Wavelet LL " + str(m))
    #     plot_3dmesh(HL, norm = "Wavelet HL " + str(m))
    #     plot_3dmesh(LH, norm = "Wavelet LH " + str(m))
    #     plot_3dmesh(HH, norm = "Wavelet HH " + str(m))
    # else:
        # plot_3dscatter(LL, m, savepath, norm = "Wavelet LL " + str(m))
    #     plot_3dscatter(HL, norm = "Wavelet HL " + str(m))
    #     plot_3dscatter(LH, norm = "Wavelet LH " + str(m))
    #     plot_3dscatter(HH, norm = "Wavelet HH " + str(m))
        

    
def normalizePlanes(nparray):
    
    data = np.zeros((1, 64, 64))

    maxHU = np.max(nparray) 
    minHU = np.min(nparray)
    
    norm = (nparray - minHU) / (maxHU - minHU)
    norm[norm>1] = 1
    norm[norm<0] = 0
    
    data[0][:][:] = norm
    
    return data

def plot_3dmesh(nparray, m, savepath, norm = "Normalized"):
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
        ax.set_xlabel("X Dimension pixels")
        ax.set_ylabel("Y Dimension pixels")
        ax.set_zlabel("Pixel Intensity")
        plt.savefig(savepath + "Size " + str(i+1), dpi = 600)
        plt.close()

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
    imagespath = r'/media/kinseylab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/*/*/*.jpg'
    resultpath = os.path.split(os.getcwd())[0] + r'/results/TumorImage/'
    img_list = glob.glob(imagespath)
    numparr = np.zeros((1,64,64))

    for n in range(len(img_list)):
        im = cv2.imread(img_list[n], 0)
        numparr[0][:][:] = im
        # cv2.imshow('images', im)

        # plot_3dmesh(numparr, norm = 'Raw')
        plot_3dmesh(normalizePlanes(numparr), 1, resultpath, norm = 'Normalized')
        
        levels = pywt.dwt_max_level(data_len = 64, filter_len = 'db1')
        print(levels)
        for m in range(levels):
            # plot_3dmesh(normalizePlanes(numparr), resultpath, norm = 'Normalized')
            dwcfilter(normalizePlanes(numparr), m, resultpath, wave = 'db1')
        
        cv2.imshow('images', im)
        PILimg_Original = Image.fromarray(im * 1.2)
        PILimg_Original = PILimg_Original.convert('RGB')
        PILimg_Original.save('Original.png', dpi = (600,600))
        plt.show()
        cv2.waitKey(0)
