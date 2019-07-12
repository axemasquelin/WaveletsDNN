'''
Project: Lung CT Wavelet Decomposition for Automated Nodule Categorization
Author: axemasquelin
Date: 10/25/2018

Function Definition: 
'''
# Libraries and Dependencies
# --------------------------------------------
from sklearn.metrics import roc_curve, auc, confusion_matrix
from preprocessing import dwcoeff, dwdec, dwcfilter
from torch.utils import data

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.tensor
import torchvision
import torch

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import random
import math
import time
import cv2, os
# --------------------------------------------

def train(trainloader, testloader, net, device, epochs, mode, lrs = 0.0001, moment = 0.99):
    
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(net.parameters(), lr= lrs, momentum= moment)
    optimizer = optim.Adam(net.parameters(), lr = lrs, betas=(0.9, 0.999), eps= 1e-8)
    #Need to change optimizer
    #lossCrit = []
    trainLoss = np.zeros((epochs))
    validLoss = np.zeros((epochs))

    for epoch in range(epochs):  # loop over the dataset multiple times
        EpochTime = 0
        running_loss = 0.0
        end = time.time()
        for i, (inputs, labels) in enumerate(trainloader):

            #Input
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(labels)

            #Outputs
            if  mode == 'Conv3':
                output, fils = net(input_var)
            else:
                output = net(input_var)

            loss = criterion(output, target_var)

            #Gradient Descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        


        EpochTime += (time.time() - end)
        print('[%d, %5d] Running loss: %.5f| Epoch Time: %.3f' %
                (epoch + 1, i + 1, running_loss/i, EpochTime))
        #lossCrit.append(running_loss/i)
        trainLoss[epoch] = running_loss/i
        validLoss[epoch] = validate(testloader, criterion, net, device, mode)
        
        running_loss = 0.0
        # if flag == 0:
        #     if (((lossCrit[epoch]- lossCrit[epoch-1])/lossCrit[epoch-1]) <= 0.1):
        #         lepoch = epoch + 1
        #         flag = 1            

    return trainLoss, validLoss

def validate(testloader, criterion, net, device, mode):
    with torch.no_grad():
        targets = np.zeros(len(testloader))
        prediction = np.transpose(np.zeros(len(testloader)))
        running_loss = 0 
        for i, (images, labels) in enumerate(testloader):
            #Load Images
            images, labels = images.to(device), labels.to(device)
            input_var = torch.autograd.Variable(images)
            target_var = torch.autograd.Variable(labels)
            
            #Input Images to Network
            if  mode == 'Conv3':
                output, fils = net(input_var)
            else:
                output = net(input_var)

            #Loss of Network
            loss = criterion(output, target_var)
            running_loss += loss.item()

    return (running_loss/i)

def test(testloader, net, device, mode = 3):
    correct = 0
    total = 0
    
    with torch.no_grad():
        # fprs, tprs = [], []
        targets = np.zeros(len(testloader))
        prediction = np.transpose(np.zeros(len(testloader)))
        count = 0
        for (images, labels) in testloader:            
            images, labels = images.to(device), labels.to(device)
            if mode == 'Conv3':
                outputs, fils = net(images)
                raw = images[0].cpu().detach().numpy()
            else:
                outputs = net(images)
                fils = 0
                raw = 0

            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

            targets[count] = labels.cpu().squeeze().numpy()
            prediction[count] = pred.cpu().squeeze().numpy()

            count += labels.size(0)
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

def init_weights(m):
    '''Initializes Model Weights using Xavier Uniform Function'''
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def weight_reset(m):
    '''Resets Model Weights'''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def set_parameter_requires_grad(model, feature_extracting = False):
    '''Freezes Model Parameters/Layers'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def calcAuc (fps, tps, mode, reps, plot_roc = True):
    ''' Calculate mean ROC/AUC for a given set of 
        true positives (tps) & false positives (fps) '''

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for itr, (_fp, _tp) in enumerate(zip(fps, tps)):
        tprs.append(np.interp(mean_fpr, _fp, _tp))
        tprs[-1][0] = 0.0
        roc_auc = auc(_fp, _tp)
        aucs.append(roc_auc)

        if plot_roc:
            plt.figure(reps, figsize=(10,8))
            plt.plot(
                _fp, _tp, lw=1, alpha=0.5,
                label='ROC fold %d (AUC = %0.2f)' % (itr+1, roc_auc)
            )
    print(len(aucs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps, mode)
        
    return aucs


def plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps, mode):
    ''' Plot roc curve per fold and mean/std score of all runs '''

    plt.figure(reps, figsize=(10,8))

    plt.plot(
        mean_fpr, mean_tpr, color='k',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc)
    )

    # plot std
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper,
        color='grey', alpha=.2, label=r'$\pm$ std.'
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('ROC Curve for ' + str(mode), fontsize=20)
    plt.legend(loc="lower right", fontsize=14)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def calcLoss_stats(loss, mode, static_fig, figure, plot_loss = True, plot_static = False):
    print("Loss: " + str(loss))
    losses = []
    
    for itr, _loss in enumerate(loss):
        print("_Loss: " + str(_loss))
        losses.append(_loss)
        
        if plot_loss == True:
            plt.figure(figure, figsize=(10,8))
            plt.plot(
                _loss, lw=1, alpha=0.5,
                label='Loss iteration %d' % (itr+1)
            )
        if plot_static == True:
            plt.figure(static_fig, figsize=(10,8))
            plt.plot(
                _loss, lw=1, alpha=0.5,
                label='Loss iteration %d' % (itr+1)
            )

    mean_loss = np.mean(losses, axis=0)
    std_loss = np.std(losses, axis=0)
    loss_upper = np.minimum(mean_loss + std_loss, 1)
    loss_lower = np.maximum(mean_loss - std_loss, 0)

    if plot_loss == True:
        plt.figure(figure)
        plt.plot(
            mean_loss, color='k',
            label=r'Mean Loss'
            )
        plt.fill_between(
            mean_loss, loss_lower, loss_upper,
            color='grey', alpha=.4, label=r'$\pm$ std.'
            )
        
        plt.title(" Loss over Epochs - " + str(mode), fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="upper right", fontsize=14)

    if plot_static == True:
        plt.figure(static_fig)
        plt.fill_between(
            mean_loss, loss_lower, loss_upper,
            color='grey', alpha=.4, label=r'$\pm$ std.'
        )
        plt.title(" Loss over Epochs - All Approaches" , fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="upper right", fontsize=14)

    return(mean_loss)

def csv_save(method, auc):
    ''' Save AUCs scores to a csv file '''

    cols = ['AUC'+str(i+1) for i in range(auc.shape[1])]
    logs = pd.DataFrame(auc, columns=cols)    
    pth_to_save =  os.path.split(os.getcwd())[0] + "/results/" + method +  "_aucs.csv"
    logs.to_csv(pth_to_save)

    print(logs)