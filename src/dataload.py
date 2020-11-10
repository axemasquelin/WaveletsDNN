# coding: utf-8
'''
    Authors:
    ---
    Axel Masquelin
    ---
'''
# Dependencies
# ---------------------------------------------------------------------------- #
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.manifold import TSNE
from sklearn.metrics import auc
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import glob, os, sys
# ---------------------------------------------------------------------------- #

def normalize_img(img):
    '''
        Description: Import dataset into a pandas dataframe and filter out empty data entires
        
    '''
    maxHU = np.max(img)                  # Find maximum pixel value of image
    minHU = np.min(img)                  # Find minimum pixel value of image
    
    norm = (img - minHU) / (maxHU - minHU)      # calculate normalized pixel values
    norm[norm>1] = 1                            # If Normal is greater than 1, set to 1
    norm[norm<0] = 0                            # if norm is less than zero set value to 0

    return norm


def resample_img(
        neg_img, pos_img,
        neg_label, pos_label,
        method
    ):
    '''
        Equalize the number of samples in each class:
        -- method = 1 - upsample the positive cases
        -- method = 2 - downsample the negative cases
    '''
    len_neg = len(neg_img)    # Benign Nodule
    len_pos = len(pos_img)    # Malignant Nodule

    # Upsample the pos samples
    if method == 1:
        pos_upsampled, pos_label = resample(
            pos_img, pos_label,
            n_samples=len_neg,
            replace=True, 
            random_state=10
        )
        neg_img = np.asarray(neg_img)
        pos_img = np.asarray(pos_img)
        neg_label = np.asarray(neg_label)
        pos_label = np.asarray(pos_label)
        return np.concatenate([pos_upsampled, neg_img]), np.concatenate([pos_label, neg_label])

    # Downsample the neg samples
    elif method == 2:
        neg_downsampled, neg_label = resample(
            neg_img, neg_label,
            n_samples=len_pos,
            replace=True, 
            random_state=10
        )
        neg_img = np.asarray(neg_img)
        pos_img = np.asarray(pos_img)
        neg_label = np.asarray(neg_label)
        pos_label = np.asarray(pos_label)
        return np.concatenate([pos_img, neg_downsampled]), np.concatenate([pos_label, neg_label])

    else:
        print('Error: unknown method')
        
def load_img(
    folder,
    normalize = True,
    resample = 2,
    ):
    '''
        Returns:
    '''
    os.chdir(folder)
    filenames = glob.glob('*.jpg')
    
    neg_img = []
    pos_img = []
    
    neg_label = []
    pos_label = []

    for i, filename in enumerate(filenames):
        im = np.zeros((1,64,64))
        ca = os.path.splitext(filename.split('_')[-1])[0]
        im[0,:,:] = np.array(Image.open(filename)) 
        
        if normalize:
            im = normalize_img(im)  
        
        if ca == '1':
            pos_img.append(im)
            pos_label.append(1) 
        else: 
            neg_img.append(im)
            neg_label.append(0)
    
    X, y = resample_img(neg_img, pos_img,
                        neg_label, pos_label,
                        resample)
    
    return X, y
