import numpy as np
import glob
import time
import pywt
import cv2
import os


def WaveCoeff (nparray, mode):
    LL_List = []
    cols = 256
    rows = 256*len(mode)

    #tifarray = tifarray[0:72,0:72,2]
    newarray = []
    coeffs = pywt.dwt2(nparray, mode)
    LL, (LH, HL, HH) = coeffs
    LL = LL[0:256, 0:256]
    newarray.append(LL)       
    LL_List.append(newarray) #Preallocate memory for variable to fix error
            
    num_images = len(LL_List)
    data = np.asarray(LL_List)
    data = data.reshape(num_images, rows, cols, 1)
    #Need to verify how this behaves in batch mode - it may not work!


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




############################
#  Multi Image Processing  #
############################
'''Defining Image Path'''


waveform = ['db1', 'db2', 'db3', 'db4']
