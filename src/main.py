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
from training_testing import *
from architectures import *

import preprocessing
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
    Definition:
    Inputs:
    Outputs:
    """
    check_gpu = torch.device("cuda:" + str(loc) if torch.cuda.is_available() else "cpu")
    print("Available Device: " + str(check_gpu))
    
    return check_gpu

def get_filters(model, raw):
    """
    Definition:
    Inputs:
    Outputs:
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
    Description:
    Input:
    Output:
    """

    if (model == "mlp"):
        size = 224
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        net = NoConv_256()
        net.apply(utils.init_weights)

    elif (model == "Wave1"):
        size = 224
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        # transforms.RandomCrop((12,12)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        net = Wave_1()
        net.apply(utils.init_weights)
    
    elif (model == "Conv1"):
        size = 224
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        # transforms.RandomCrop((12,12)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        net = Conv_1()
        net.apply(utils.init_weights)

    elif (model == "inception_wave"):
        size = 224
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        # transforms.RandomCrop((12,12)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        net = incept_wave()
        net.apply(utils.init_weights)

    elif (model == "inception_conv"):
        size = 224
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        # transforms.RandomCrop((12,12)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        net = incept_conv()
        net.apply(utils.init_weights)

    elif model == "AlexNet":
        size = 224
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        # transforms.Grayscale(),
                                        transforms.ToTensor()])
        net = AlexNet()
        # net.apply(utils.init_weights)

    elif model == "WalexNet":
        size = 224
        transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Grayscale(),
                                        transforms.ToTensor()])
        net = WalexNet()

    elif model == "alexnet":
        size = 224
        transform = transforms.Compose([transforms.Resize(size, interpolation = 2),
                                        transforms.RandomHorizontalFlip(p=0.8),
                                        transforms.ToTensor()
                                        ])
        net = torchvision.models.alexnet(pretrained = True)
        utils.set_parameter_requires_grad(net,feature_extracting = True)
        net.classifier[1] = nn.Linear(9216, 50)
        net.classifier[4] = nn.Linear(50, 10)
        net.classifier[6] = nn.Linear(10, 2)
    
    else:
        print("Warning: Model Not Found")
    
    return net, transform

if __name__ == '__main__':
    """
    Definition:
    Inputs:
    Outputs:
    """

    # Network Parameters
    models = [
            'Wave1',            # Single Level Wavelet Decomposition Layer extracting 4 features
            'Conv1',            # Convolutional Layer 4 Feature Extracted
            'inception_wave',   # Multi Level Wavelet Decomposition
            'inception_conv',   # Multiscale Convolutional Module.
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
    class_names = ["Malignant", "Benign"]   # Class Name (0 - Malignant, 1 - Benign)
    gpu_loc = 0                             # Define GPU to use (0 or 1)
    seed = 2020                             # Define Random Seed
    reps = 50                                # Define number of repetition for each test
    fig = 3                                 # Defines figure counter
    
    # GPU Initialization
    device = GPU_init(gpu_loc)

    # Defining Dataset Path
    TrainPath = r'/media/kinseylab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group1/'
    TestPath = r'/media/kinseylab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group0/'    
        
    for model in models:
        print("Selected Model: " + str(model))
        np.random.seed(seed)  

        # Initialize Network and send it to GPU
        net, transform = net_select(model)
        net = net.to(device)
        
        # Define Static figure counters
        static_fig = 0
        mean_training_fig = 1
        mean_valid_fig = 2

        # Defining empty lists to store network performance information
        fprs, tprs, trainloss, valloss = [], [], [], []
        auc_scores = np.zeros((1,reps))

        for r in range(reps):
            progressBar(r + 1, reps)

            # Load Training datasets 
            trainset = torchvision.datasets.ImageFolder(TrainPath, transform = transform)    
            trainloader = torch.utils.data.DataLoader(trainset, batch_size= 100, shuffle= True)
            
            # Load Testing Datasets
            testset = torchvision.datasets.ImageFolder(TestPath, transform = transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size= 100, shuffle= True)

            trainLoss, validLoss, trainAcc, validAcc = train(trainloader, testloader, net, device, r,  opt, model)
            
            trainloss.append(trainLoss)
            valloss.append(validLoss)
            
            utils.plot_losses(fig, trainLoss, validLoss, model)
            utils.plot_accuracies(fig, trainAcc, validAcc, model)

                            
            confmatrix, fp, tp, fils, raw = test(testloader, net, device, mode = model)
            
            fprs.append(fp), tprs.append(tp)

            net.apply(utils.weight_reset)
    
            
            plt.figure(fig)
            utils.plot_confusion_matrix(confmatrix, class_names, r, model, normalize = True, title = 'Normalize Confusion Matrix ' + str(model))
            fig += 1
        
        
        print('fprs: ' + str(fprs))
        print('tprs: ' + str(tprs))
        auc_scores[0,:] = utils.calcAuc(fprs,tprs, model, fig, plot_roc= True)
        fig += 1

        mean_losses, loss_upper, loss_lower = utils.calcLoss_stats(trainloss, model, static_fig, fig, plot_loss = True, plot_static= True)
        fig += 1

        print("Length of Mean: " + str(len(mean_losses)))
        print("Length of Upper: " + str(len(loss_upper)))
        print("Length of Lower: " + str(len(loss_lower)))

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

        print("Length of Mean: " + str(len(mean_losses)))
        print("Length of Upper: " + str(len(loss_upper)))
        print("Length of Lower: " + str(len(loss_lower)))

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


        utils.csv_save(model, auc_scores)             

            