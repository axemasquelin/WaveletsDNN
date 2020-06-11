
'''
Project: Lung CT Wavelet Decomposition for Automated Nodule Categorization
Author: axemasquelin
Date: 10/25/2018
 
'''
# Libraries and Dependencies
# --------------------------------------------
from sklearn.metrics import roc_curve, auc, confusion_matrix
from preprocessing import *
from torch.utils import data
from loss import FocalLoss


import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.tensor
import torchvision
import torch

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import utils
import math
import time
import cv2, os
# --------------------------------------------

def optim_criterion_select(net, opt):
    """
    Description:
    Input:
    Output:
    """
    # Define Loss Functions
    if opt['loss'] == 'focal':
        crit = FocalLoss().cuda()
    if opt['loss'] == 'entropy':
        crit = nn.CrossEntropyLoss().cuda()

    # Define Optimizer
    if opt['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = opt['lr'], betas= opt['betas'], eps= opt['eps'])
    if opt['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr= opt['lr'], momentum= opt['momentum'])
    if opt['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr = opt['lr'], rho = opt['rho'], eps = opt['eps'], weight_decay = opt['decay'])

    return crit, optimizer

def train(trainloader, testloader, net, device, rep, opt, model):
    """
    Description:
    Input:
    Output:
    """

    criterion, optimizer = optim_criterion_select(net, opt)

    trainLoss = np.zeros((opt['epchs']))    # Defining Zero Array of Training Loss
    validLoss = np.zeros((opt['epchs']))    # Defining Zero Array for Validation Loss
    trainAcc = np.zeros((opt['epchs']))     # Defining Zero Array for Training Accuracy
    validAcc = np.zeros((opt['epchs']))     # Defining Zero Array for Validation Accuracy

    for epoch in range(opt['epchs']):  # loop over the dataset multiple times

        EpochTime = 0       # Zeroing Epoch Timer
        running_loss = 0.0  # Zeroing Running Loss per epoch
        total = 0           # Zeroing total images processed
        correct = 0         # Zeroing total classes correct

        # Updating Learning Rate based on epoch
        # utils.adjust_lr(optimizer, opt['lr'], opt['epchs'])    

        end = time.time()
        for i, (images, labels) in enumerate(trainloader):
            # print(images.size())
            # Input
            images = images.to(device)
            labels = labels.to(device)
            images = torch.autograd.Variable(images)
            labels = torch.autograd.Variable(labels)
            
            optimizer.zero_grad()
            
            output = net(images, device)
            _, predicted = torch.max(output,1)
            loss = criterion(output, labels)

            # Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Gradient Descent
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        trainLoss[epoch], trainAcc[epoch] = running_loss/i, (correct/total) * 100 
        validLoss[epoch], validAcc[epoch] = validate(testloader, criterion, net, device, model)

        EpochTime += (time.time() - end)
        print('[Mode: %s, Rep: %i, Epoch: %d, Epoch Time: %.3f]' %(model, rep, epoch + 1, EpochTime))
        print('Train loss: %.5f | Train Accuracy: %.5f' %(trainLoss[epoch], trainAcc[epoch]))
        print('Valid loss: %.5f | Valid Accuracy: %.5f \n' %( validLoss[epoch], validAcc[epoch]))
       
        running_loss = 0.0

    return trainLoss, validLoss, trainAcc, validAcc

def validate(testloader, criterion, net, device, mode):
    """
    Description:
    Input:
    Output:
    """

    with torch.no_grad():
        running_loss = 0 
        total = 0
        correct = 0

        for i, (images, labels) in enumerate(testloader):

            # Load Images
            images, labels = images.to(device), labels.to(device)
            input_var = torch.autograd.Variable(images)
            target_var = torch.autograd.Variable(labels)
            
            output = net(input_var, device)

            _, predicted = torch.max(output,1)
            loss = criterion(output, target_var)

            # Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()
        

    return (running_loss/i, correct/total * 100)

def test(testloader, net, device, mode = 3):
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        targets = []        # np.zeros(len(testloader))
        prediction = []     # np.transpose(np.zeros(len(testloader)))
        softpred = []
        count = 0

        for (images, labels) in testloader:

            images, labels = images.to(device), labels.to(device)

            outputs = net(images, device)                                                
            fils = 0
            raw = 0

            _, pred = torch.max(outputs,1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

            for i in range(len(labels)):
                targets.append(labels[i].cpu().squeeze().numpy())
                prediction.append(pred[i].cpu().squeeze().numpy())
                softpred.append(outputs[i,1].cpu().squeeze().numpy())
                count += 1

            
        acc = (100 * correct/total)
        print('Accuracy of Network: %d %%' %acc)

        fp, tp, threshold = roc_curve(targets, softpred[:])
        print('FP: ' + str(fp))
        print('TP: ' + str(tp))
        
        conf_matrix = confusion_matrix(prediction, targets)
    
    return (
        conf_matrix,
        fp,
        tp,
        fils,
        raw
    )
