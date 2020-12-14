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
# ---------------------------------------------------------------------------- #

def tensor_cat(x1 = None, x2 = None, x3 = None, x4 = None, x5 = None, x6 = None, padding = True): #Explain this function
    '''Concatenate Different Size Features, zero padding to match size ''' 
    if padding:            # Adding Padding around feature outputes. - Sparsity could impact performance. 
        if x2 is not None:
            # print(x1.size())
            # print(x2.size())
            x2pad = F.pad(input = x2, pad=(8,8,8,8), mode = 'constant', value = 0)
            xcat = torch.cat((x1,x2pad),1)
        if x3 is not None:
            x3pad = F.pad(input = x3, pad=(12,12,12,12), mode = 'constant', value = 0)
            xcat = torch.cat((xcat,x3pad),1)
        if x4 is not None:
            x4pad = F.pad(input = x4, pad=(14,14,14,14), mode = 'constant', value = 0)
            xcat = torch.cat((xcat,x4pad),1)
        if x5 is not None:
            x5pad = F.pad(input = x5, pad=(15,15,15,15), mode = 'constant', value = 0)
            xcat = torch.cat((xcat,x5pad),1)
        if x6 is not None:
            x6pad = F.pad(input = x6, pad=(15,16,15,16), mode = 'constant', value = 0)
            xcat = torch.cat((xcat,x6pad),1)
    
    else:                   # Impact of Resizing to 32x32 without padding - does sparsity impact performance?    
        if x2 is not None:
            x2scaled = F.interpolate(x2, size = 32) 
            xcat = torch.cat((x1,x2scaled),1)
        if x3 is not None:
            x3scaled = F.interpolate(x3, size = 32) 
            xcat = torch.cat((xcat,x3scaled),1)
        if x4 is not None:
            x4scaled = F.interpolate(x4, size = 32) 
            xcat = torch.cat((xcat,x4scaled),1)
        if x5 is not None:
            x5scaled = F.interpolate(x5, size = 32) 
            xcat = torch.cat((xcat,x5scaled),1)
        if x6 is not None:
            x6scaled = F.interpolate(x6, size = 32) 
            xcat = torch.cat((xcat,x6scaled),1)
    
    return xcat

def adjust_lr(optimizer, lrs, epoch):
    lr = lrs * (0.01 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_weights(m):
    '''Initializes Model Weights using Xavier Uniform Function'''
    np.random.seed(2019)
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


def calcAuc (fps, tps, mode, reps, plot_roc = False):
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
                # label='ROC fold %d (AUC = %0.2f)' % (itr+1, roc_auc)
            )
    print(len(aucs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    if plot_roc:
        plot_roc_curve(tprs, mean_fpr, mean_tpr, mean_auc, std_auc, reps, mode)
    print(aucs)
    return aucs


def plot_losses(fig, trainLoss, validLoss, mode):
    plt.figure(fig)
    plt.plot(trainLoss)
    plt.plot(validLoss)
    plt.title('Loss over Epochs ' + str(mode))
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Error)')
    plt.legend(['Training','Validation'], loc = 'top right')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(mode) + '_Loss.png', dpi = 600)
    plt.close()


def plot_accuracies(fig, trainAcc, validAcc, mode):
    plt.figure(fig)
    plt.plot(trainAcc)
    plt.plot(validAcc)
    plt.title('Accuracies over Epochs ' + str(mode))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training','Validation'], loc = 'top right')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(mode) + '_Accuracies.png', dpi = 600)
    plt.close()


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
        color='grey', alpha=.4, label=r'$\pm$ std.'
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title('ROC Curve for ' + str(mode), fontsize=16)
    plt.legend(loc="lower right", fontsize=14)
    savepath = os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '/' + str(mode) + '_ROC.png'
    plt.savefig(savepath, dpi=600)
    plt.close()


def plot_confusion_matrix(cm, classes, r, model,
                          normalize=False,
                          title='Confusion matrix',
                          saveFlag = False,
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

    tick_marks = np.arange(len(classes))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
  
    plt.colorbar()

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    if saveFlag:
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        savepath = os.path.split(os.getcwd())[0] + '/results/' + str(model) + '/' + str(model) + "_best_model.png"
        plt.savefig(savepath, dpi = 600)
        plt.close()
  


def calcLoss_stats(loss, mode, static_fig, figure, plot_loss = True, plot_static = False):
    # print("Loss: " + str(loss))
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
            np.arange(0,len(mean_loss)), loss_lower, loss_upper,
            alpha=.2, label=r'$\pm$ std.'
            )
        
        plt.title(" Loss over Epochs - " + str(mode), fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(loc="upper right", fontsize=14)

    if plot_static == True:
        plt.figure(static_fig)
        plt.fill_between(
            np.arange(0,len(mean_loss)), loss_lower, loss_upper,
            alpha=.3, label=r'$\pm$ std.'
        )
        plt.title(" Loss over Epochs - All Approaches" , fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.legend(loc="upper right", fontsize=14)

    return mean_loss, loss_upper, loss_lower

def model_save(method, net):
    print("Saving Network")
    net_path = os.path.split(os.getcwd())[0] + "/results/" + method + '/' + method + '_bestnetwork.pt'
    torch.save(net, net_path)

def csv_save(method, data, name = ''):
    ''' Save AUCs scores to a csv file '''

    cols = [name +str(i+1) for i in range(data.shape[1])]
    logs = pd.DataFrame(data, columns=cols)    
    pth_to_save =  os.path.split(os.getcwd())[0] + "/results/" + method + '/' + method +  "_" + name + ".csv"
    logs.to_csv(pth_to_save)

    print(logs)