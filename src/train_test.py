
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
from utils import adjust_lr

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
import math
import time
import cv2, os
# --------------------------------------------

def train(trainloader, testloader, net, device, rep, epochs, mode, lrs = 0.001, moment = 0.9):

    # criterion = FocalLoss().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(net.parameters(), lr= lrs, momentum= moment)
    optimizer = optim.Adam(net.parameters(), lr = lrs, betas=(0.9, 0.999), eps= 1e-8)
    # optimizer = optim.Adadelta(net.parameters(), lr = lrs, rho = 0.9, eps = 1e-6, weight_decay = 0.001)

    trainLoss = np.zeros((epochs))
    validLoss = np.zeros((epochs))
    trainAcc = np.zeros((epochs))
    validAcc = np.zeros((epochs))

    for epoch in range(epochs):  # loop over the dataset multiple times
        adjust_lr(optimizer, lrs, epoch)    
        EpochTime = 0
        running_loss = 0.0
        total = 0
        correct = 0
        
        end = time.time()
        for i, (images, labels) in enumerate(trainloader):
            if mode == 'WaveCon':
                images = dwcfilter(images, wave = 'db1')
            elif mode == 'WaveS':
                images = dwdec(images, wave = 'db1')       
            elif mode == 'Raw':
                images = normalizePlanes(images)
        
            #Input
            images = images.to(device)
            labels = labels.to(device)
            images = torch.autograd.Variable(images)
            labels = torch.autograd.Variable(labels)

            #Outputs
            if  mode == 'Conv3':
                output, fils = net(images)
            else:
                output = net(images)

            predicted = torch.max(output[0],1)
            loss = criterion(output[0], labels)

            #Accuracy
            total += labels.size(0)
            correct += (predicted[1] == labels).sum().item()

            #Gradient Descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        
        trainLoss[epoch], trainAcc[epoch] = running_loss/i, (correct/total) * 100 
        validLoss[epoch], validAcc[epoch] = validate(testloader, criterion, net, device, mode)

        EpochTime += (time.time() - end)
        print('[Mode: %s, Rep: %i, Epoch: %d, Epoch Time: %.3f]' %(mode, rep, epoch + 1, EpochTime))
        print('Train loss: %.5f | Train Accuracy: %.5f' %(trainLoss[epoch], trainAcc[epoch]))
        print('Valid loss: %.5f | Valid Accuracy: %.5f \n' %( validLoss[epoch], validAcc[epoch]))
       
        running_loss = 0.0

    return trainLoss, validLoss, trainAcc, validAcc

def validate(testloader, criterion, net, device, mode):
    with torch.no_grad():
        running_loss = 0 
        total = 0
        correct = 0

        for i, (images, labels) in enumerate(testloader):
            if mode == 'WaveCon':
                images = dwcfilter(images, wave = 'db1')         
            elif mode == 'WaveS':
                images = dwdec(images, wave = 'db1')
            elif mode == 'Raw':
                images = normalizePlanes(images)

            #Load Images
            images, labels = images.to(device), labels.to(device)
            input_var = torch.autograd.Variable(images)
            target_var = torch.autograd.Variable(labels)
            
            #Input Images to Network
            if  mode == 'Conv3':
                output, fils = net(input_var)
            else:
                output = net(input_var)

            predicted = torch.max(output[0],1)

            #Loss of Network
            loss = criterion(output[0], target_var)

            #Accuracy
            total += labels.size(0)
            correct += (predicted[1] == labels).sum().item()

            running_loss += loss.item()
        

    return (running_loss/i, correct/total * 100)

def test(testloader, net, device, mode = 3):
    correct = 0
    total = 0
    
    with torch.no_grad():
        # fprs, tprs = [], []
        targets = [] #np.zeros(len(testloader))
        prediction = [] #np.transpose(np.zeros(len(testloader)))
        count = 0
        for (images, labels) in testloader:
            if mode == 'WaveCon':
                images = dwcfilter(images, wave = 'db1')         
            elif mode == 'WaveS':
                images = dwdec(images, wave = 'db1')
            elif mode == 'Raw':
                images = normalizePlanes(images)           
            
            images, labels = images.to(device), labels.to(device)
            
            if mode == 'Conv3':
                outputs, fils = net(images)
                raw = images[0].cpu().detach().numpy()
            else:
                outputs = net(images)
                fils = 0
                raw = 0

            pred = torch.max(outputs[0],1)

            total += labels.size(0)
            correct += (pred[1] == labels).sum().item()

            for i in range(len(labels)):
                targets.append(labels[i].cpu().squeeze().numpy())
                prediction.append(pred[1][i].cpu().squeeze().numpy())
                count += 1

            
        acc = (100 * correct/total)
        print('Accuracy of Network: %d %%' %acc)

        fp, tp, threshold = roc_curve(targets, prediction[:])
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
