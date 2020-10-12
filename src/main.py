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
from sklearn.model_selection import train_test_split
from skorch.dataset import Dataset
from training_testing import *
from architectures import *

import preprocessing
import dataload
import utils

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.tensor
import torchvision
import torch

import matplotlib.pyplot as plt
import numpy as np
import cv2, sys, os
# ---------------------------------------------------------------------------- #

def progressBar(value, endvalue, bar_length=50):
    """
    Definition:
    Inputs:
    Outputs:
    """
    percent = float(value) / endvalue
    arrow = '=' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\n\r[{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    print('\n')

def GPU_init(loc):
    """
    Definition: GPU Initialization function
    Inputs: loc - 0 or 1 depending on which GPU is being utilized
    Outputs: check_gpu - gpu enabled variable
    """
    check_gpu = torch.device("cuda:" + str(loc) if torch.cuda.is_available() else "cpu")
    print("Available Device: " + str(check_gpu))
    
    return check_gpu

def get_filters(model, raw, fils):
    """
    Definition: Visualization function to show what each filter was focusing on for Conv1
    Inputs: model - string of model
    """
    
    # Raw Image
    cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + model + "_Original.png", raw[0,:,:]*255)
    preprocessing.pywtFilters(raw, 'db1')
    
    # Convolution Activation
    img = np.zeros((128,128,3))
    img[:,:,0] = fils[0,0,:,:]
    img[:,:,1] = fils[0,1,:,:]
    img[:,:,2] = fils[0,2,:,:]
    
    # Convert this into a loop
    cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + model + "_Allfilter.png", img*1000)
    cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + model + "_filter01.png", fils[0,0,:,:]*1000)
    cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + model + "_filter02.png", fils[0,1,:,:]*1000)
    cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + model + "_filter03.png", fils[0,2,:,:]*1000)

def net_select(model):
    """
    Description: Network selection function
    Input: model - string that defines which model will be used
    Output: net - loaded network
    """
    
    if (model == "Conv1"):
        net = Conv_1()
        net.apply(utils.init_weights)
    
    elif (model == "Conv2"):
        net = incept_conv()
        net.apply(utils.init_weights)

    elif (model == "Wave1"):
        net = Wave_1()
        net.apply(utils.init_weights)
    elif (model == "Wave2"):
        net = incept_wave2()
        net.apply(utils.init_weights)

    elif (model == "Wave3"):
        net = incept_wave3()
        net.apply(utils.init_weights)
    
    elif (model == "Wave4"):
        net = incept_wave4()
        net.apply(utils.init_weights)

    elif (model == "Wave5"):
        net = incept_wave5()
        net.apply(utils.init_weights)

    elif (model == "Wave6"):
        net = incept_wave6()
        net.apply(utils.init_weights)    
    
    else:
        print("Warning: Model Not Found")
    
    return net

if __name__ == '__main__':
    """
    Definition:
    Inputs:
    Outputs:
    """

    # Network Parameters
    models = [
            'Wave1',   # Single Level Wavelet Decomposition Layer extracting 4 features
            # 'Wave2',   # Multi Level Wavelet Decomposition
            # 'Wave3',   # Multi Level Wavelet Decomposition
            # 'Wave4',   # Multi Level Wavelet Decomposition
            # 'Wave5',   # Multi Level Wavelet Decomposition
            # 'Wave6',   # Multi Level Wavelet Decomposition
            'Conv1',   # Convolutional Layer 4 Feature Extracted
            # 'Conv2',   # Multiscale Convolutional Module.

            # 'AlexNet',         # Standard Alexnet Architecture with modified classifier
            # 'WalexNet',        # Wavelet Alexnet Architecture
            ]

    wkernel = [
            'db1',
            'db2',
            'sym1',
            'gaus',
    ]

    opt = {
            'loss': 'entropy',          # entropy or focal
            'optimizer': 'Adam',        # SGD, Adam, or Adadelta
            'epchs': 125,               # Number of Epochs
            'lr': 0.001,                 # Learning Rate
            'betas': (0.9, 0.999),      # Beta parameters for Adam optimizer
            'rho': 0.9,                 # rho parameter for adadelta optimizer
            'eps': 1e-7,                # epsilon paremeter for Adam and Adadelta optimzier
            'decay': 0.001,             # Decay rate for Adadelta optimizer
            'momentum': 0.99            # Momentum parameter for SGD optimizer
        }
    
    # Flags
    check_feats = True  # Check Features from Convolution Layers
    check_grad = True   # check
    save_figs = True    # Checks to save figures to /results/
    save_auc = True     # Checks to save .csv of all AUCs

    # Variables
    class_names = ["Benign", "Malignant"]   # Class Name (1 - Malignant, 0 - Benign)
    gpu_loc = 0                             # Define GPU to use (0 or 1)
    seed = 2019                             # Define Random Seed
    folds = 5                               # Cross Validation Folds
    reps = 5                               # Define number of repetition for each test
    fig = 3                                 # Defines figure counter
    
    # GPU Initialization
    device = GPU_init(gpu_loc)

    # Defining Dataset Path
    # TrainPath = r'/media/kinseylab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group1/'
    # TestPath = r'/media/kinseylab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group0/'    
    allfiles =r'/media/kinseylab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group2/'
    
    cwd = os.getcwd()
    X, y = dataload.load_img(allfiles)
    os.chdir(cwd)

    for model in models:
        print("Selected Model: " + str(model))
        np.random.seed(seed)  

        # Initialize Network and send it to GPU
        net = net_select(model)
        net = net.to(device)

        pytorch_total_params = sum(p.numel() for p in net.parameters())
        # print(pytorch_total_params)
        
        # Define Static figure counters
        static_fig = 0
        mean_training_fig = 1
        mean_valid_fig = 2

        # Defining empty lists to store network performance information
        trainloss, valloss =  [], []
        trainacc, valacc =  [], []
        sensitivity =  np.zeros((folds,reps))
        specificity =  np.zeros((folds,reps))
        auc_scores = np.zeros((folds,reps))
        train_time = np.zeros((folds,reps))
        best_acc = 0
        for k in range(folds):
            progressBar(k + 1, reps)
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = k)
            fprs, tprs = [], []

            for r in range(reps):

                # Load Training Dataset
                trainset = Dataset(X_train, y_train)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size= 100, shuffle= True)

                #Load testing Dataset
                testset = Dataset(X_train, y_train)
                testloader = torch.utils.data.DataLoader(testset, batch_size= 100, shuffle= True)

                trainLoss, validLoss, trainAcc, validAcc, trainTime = train(trainloader, testloader, net, device, r,  opt, model)
                
                trainloss.append(trainLoss)
                valloss.append(validLoss)
                trainacc.append(trainAcc)
                valacc.append(validAcc)
                train_time[k,r] = trainTime

                utils.plot_losses(fig, trainLoss, validLoss, model)
                utils.plot_accuracies(fig, trainAcc, validAcc, model)
                
                print('[Mode: %s, Fold: %i, Rep: %i]' %(model, k, r))
                confmatrix, fp, tp, sens, spec, fils, acc, raw = test(testloader, net, device, mode = model)             
                
                if acc > best_acc:
                    best_acc = acc
                    utils.model_save(model, net)
                    
                    plt.figure(fig)
                    utils.plot_confusion_matrix(confmatrix, class_names, r, model, normalize = True, title = 'Normalize Confusion Matrix ' + str(model), saveFlag = True)
                    fig += 1
                
                
                sensitivity[k,r] = sens
                specificity[k,r] = spec
                
                
                fprs.append(fp), tprs.append(tp)
                net.apply(utils.weight_reset)

            auc_scores[k,:] = utils.calcAuc(fprs,tprs, model, fig, plot_roc= True)
            fig += 1
        
        mean_losses, loss_upper, loss_lower = utils.calcLoss_stats(trainloss, model, static_fig, fig, plot_loss = True, plot_static= True)
        fig += 1

        plt.figure(mean_training_fig)
        plt.plot(mean_losses)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(" Average Training Loss over Epochs")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(models, loc = 'top right')
    
    
        plt.fill_between(
            np.arange(0,opt['epchs']), loss_lower, loss_upper,
            alpha=.2, label=r'$\pm$ std.'
            )
        savepath = os.path.split(os.getcwd())[0] + '/results/AllApproaches_AverageLoss.png'
        plt.savefig(savepath, dpi = 100)

        #Validation Loss
        mean_losses, loss_upper, loss_lower = utils.calcLoss_stats(valloss, model, static_fig, fig, plot_loss = True, plot_static= True)
        fig += 1

        # print("Length of Mean: " + str(len(mean_losses)))
        # print("Length of Upper: " + str(len(loss_upper)))
        # print("Length of Lower: " + str(len(loss_lower)))

        plt.figure(mean_valid_fig)
        plt.plot(mean_losses)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(" Average Validation Loss over Epochs")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(models, loc = 'top right')
    
    
        plt.fill_between(
            np.arange(0,opt['epchs']), loss_lower, loss_upper,
            alpha=.2, label=r'$\pm$ std.'
            )
        savepath = os.path.split(os.getcwd())[0] + '/results/AllApproaches_AverageValidationLoss.png'
        plt.savefig(savepath, dpi = 100)
        
        utils.csv_save(model, train_time, name = 'time')
        utils.csv_save(model, sensitivity, name = 'sensitivity')
        utils.csv_save(model, specificity, name = 'specificity')
        utils.csv_save(model, auc_scores, name = 'auc')             

        