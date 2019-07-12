'''
Project: Lung CT Wavelet Decomposition for Automated Nodule Categorization
Author: Axel Masquelin
Date: 10/25/2018

'''
####################################
import torch.tensor
import torchvision
import torch

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pywt
import types, time
import glob, cv2, os
####################################

'''Normalize Grayscale'''
def normalizePlanes(nparray):
    maxHU = np.max(nparray) 
    minHU = np.min(nparray)
    
    nparray = (nparray - minHU) / (maxHU - minHU)
    nparray[nparray>1] = 1
    nparray[nparray<0] = 0
    return nparray


'''Discrete Wavelet Deconstruction'''
def dwcoeff (loader, wave):
    
    numparr = loader.data.cpu().numpy()

    LL_arr = np.zeros((len(loader), 1, 256,256))
    for i in range(len(loader)):   
        coeffs = pywt.dwt2(normalizePlanes(numparr[i]), wave)
        LL, (LH, HL, HH) = coeffs
        temp1 = np.concatenate((LL,LH*5), axis = 1)
        temp2 = np.concatenate((HL*5,HH*5), axis = 1)

        bp = normalizePlanes(np.concatenate((temp1,temp2), axis = 2))
        
        LL_arr[i][0][:][:] = bp

        # cv2.imshow('db1', LL_arr[i][0][:][:])
        # cv2.waitKey(0)
        

    data = np.asarray(LL_arr)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    return data

'''Discrete Wavelet Deconstruction Scalogram'''
def dwdec (loader, wave):
    shape = (256, 256)
    max_lev = 5
    numparr = loader.data.cpu().numpy()
    WaveDec = np.zeros((len(loader), 1, 256, 256), dtype = float)
     
    for i in range(len(loader)):
        initflag = 1
        cv2.imshow('Original', numparr[i][0])
        coeffs = pywt.wavedec2(numparr[i], wave)
        
        for n in range(len(coeffs)-1):
            print()
            print("Coeffs @" + str(n) + " " + str(coeffs[n]))
            if initflag == 0:
                temp1 = np.concatenate((imgScale, coeffs[n+1][0][0]), axis = 1)
                temp2 = np.concatenate((coeffs[n+1][1][0], coeffs[n+1][2][0]), axis = 1)
                print("Temp 1: " + str(temp1))
                print("Temp 2: " + str(temp2))
                imgScale = np.concatenate((temp1, temp2), axis = 0)
                
            else:
                temp1 = np.concatenate((coeffs[n][0], coeffs[n+1][0][0]), axis = 1)
                temp2 = np.concatenate((coeffs[n+1][1][0], coeffs[n+1][2][0]), axis = 1)
                print("Temp 1: " + str(temp1))
                print("Temp 2: " + str(temp2))
                imgScale = np.concatenate((temp1, temp2), axis = 0)
                initflag = 0

        WaveDec[i][:][:][:] = imgScale
        


    data = np.asarray(WaveDec)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    #print(data.size())

    return data

'''Discrete Wavelet Deconstruction 3D Filter'''
def dwcfilter (loader, wave):
    
    numparr = loader.data.cpu().numpy()
    
    LL_arr = np.zeros((len(loader), 4, 128,128))
    for i in range(len(loader)):   
        random = []
        imageShow = []
        coeffs = pywt.dwt2(numparr[i], wave)
        LL, (LH, HL, HH) = coeffs
        LL = np.concatenate((LL,LH,HL,HH), axis = 0)
        
        LL_arr[i][:][:][:] = LL

    data = np.asarray(LL_arr)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    return data


def pywtFilters (loader, wave):
    
    numparr = loader#.data.cpu().numpy()
    
    LL_arr = np.zeros((len(loader), 4, 128,128))
    for i in range(len(loader)):   
        random = []
        imageShow = []
        coeffs = pywt.dwt2(numparr[i], wave)
        LL, (LH, HL, HH) = coeffs
        cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/HH.png", HH*1000)
        cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/LH.png", LH*1000)
        cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/HL.png", HL*1000)
        LL = np.concatenate((LL,LH,HL,HH), axis = 0)


'''Generating Wavelet Reconstruction Images'''
def Reconstruction(nparray, mode='haar'):
    #Compute Transform Coefficients
    coeffs = pywt.wavedec2(nparray, mode)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0.25 #Filtering

    #Wavelet Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = imArray_H * 255
    imArray_H = np.uint8(imArray_H)
    return(imArray_H)
