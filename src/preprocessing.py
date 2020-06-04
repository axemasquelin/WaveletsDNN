# coding: utf-8
""" MIT License """
'''
    Project: Wavelet DNN
    Authors: Axel Masquelin
    Description: Image preprocessing scripts to visualize CT images, normalize the image,
    and conduct multiscale wavelet decomposition
'''
# ---------------------------------------------------------------------------- #
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.tensor
import torchvision
import torch
import pywt
import glob, cv2, os, types, time
# ---------------------------------------------------------------------------- #

def normalizePlanes(loader):
    """
    Description:
    Input:
    Output:
    """

    nparray = loader.data.cpu().numpy()             # Convert loader data into numpy array
    data = np.zeros((len(loader), 1, 64, 64))       # Create a new matrix of zeroes

    for i in range(len(loader)):                    # Loop over length of all images
        maxHU = np.max(nparray[i])                  # Find maximum pixel value of image
        minHU = np.min(nparray[i])                  # Find minimum pixel value of image
        
        norm = (nparray[i] - minHU) / (maxHU - minHU)   # calculate normalized pixel values
        norm[norm>1] = 1                                # If Normal is greater than 1, set to 1
        norm[norm<0] = 0                                # if norm is less than zero set value to 0
        
        data[i][0][:][:] = norm                     # Load normalized image into new loader structure
    
    data = np.asarray(data)                         # Define data as an array
    data = torch.from_numpy(data)                   # Convert numpy array to tensor structure
    data = data.type(torch.FloatTensor)             # Convert to floaters

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


def dwdec (loader, wave):
    """
    Description:
    Input:
    Output:
    """
    shape = (128, 128)
    numparr = loader.data.cpu().numpy()
    WaveDec = np.zeros((len(loader), 1, 64, 64), dtype = float)
     
    for i in range(len(loader)):
        initflag = 1
        levels = pywt.dwt_max_level(data_len = 64, filter_len = wave)
        coeffs = pywt.wavedec2(numparr[i], wave)

        for n in range(len(coeffs)-1):

            if initflag == 0:
                temp1 = np.concatenate((imgScale, coeffs[n+1][0][0]), axis = 1)
                temp2 = np.concatenate((coeffs[n+1][1][0], coeffs[n+1][2][0]), axis = 1)

                imgScale = np.concatenate((temp1, temp2), axis = 0)
                
            else:
                temp1 = np.concatenate((coeffs[n][0], coeffs[n+1][0][0]), axis = 1)
                temp2 = np.concatenate((coeffs[n+1][1][0], coeffs[n+1][2][0]), axis = 1)

                imgScale = np.concatenate((temp1, temp2), axis = 0)
                initflag = 0

        WaveDec[i][:][:][:] = imgScale
    
    data = np.asarray(WaveDec)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)

    return data

def singlelvl_wd(loader, device, size):
    """
    Description: Single Level Wavelet Decomposition for  Wavelet Layer in NN Architecture
    Input:      (1) Loader - images loaded to GPU/device
                (2) device - location of memory (GPU/CPU)
                (3) size   - Size of images that will be generated Half of input image
    Output:     (1) data   - Images to be sent to network
    """

    numparr = loader.data.cpu().numpy()
    wavelet_img = np.zeros((len(loader), 4, size, size), dtype = float)

    for i in range(len(loader)):
        coeffs = pywt.dwt2(numparr[i], 'db1')
        LL, (LH, HL, HH) = coeffs

        wavelet_img[i,0,:,:] = LL
        wavelet_img[i,1,:,:] = LH
        wavelet_img[i,2,:,:] = HL
        wavelet_img[i,3,:,:] = HH
    
    data = np.asarray(wavelet_img)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    data = data.to(device)

    return data


def multiscale_wd(loader, device, size):
    """
    Description:
    Input:
    Output:
    """

    numparr = loader.data.cpu().numpy()
    LL_array = np.zeros((len(loader), 1, size, size), dtype = float)
    wavelet_img = np.zeros((len(loader), 3, size, size), dtype = float)

    for i in range(len(loader)):
        coeffs = pywt.dwt2(numparr[i], 'db1')
        LL, (LH, HL, HH) = coeffs

        LL_array[i,:,:,:] = LL
        wavelet_img[i,0,:,:] = LH
        wavelet_img[i,1,:,:] = HL
        wavelet_img[i,2,:,:] = HH
    
    data = np.asarray(wavelet_img)
    data = torch.from_numpy(data)
    data = data.type(torch.FloatTensor)
    data = data.to(device)

    LL_array = np.asarray(LL_array)
    LL_array = torch.from_numpy(LL_array)
    LL_array = LL_array.type(torch.FloatTensor)
    LL_array = LL_array.to(device)


    return LL_array, data