'''
Project: Lung CT Wavelet Decomposition for Automated Nodule Categorization
Author: axemasquelin
Date: 10/25/2018
 
'''
# Libraries and Dependencies
# --------------------------------------------
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
# --------------------------------------------

def adjust_lr(optimizer, lrs, epoch):
    lr = lrs * (0.1 ** (epoch // 20))
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

def plot_losses(fig, trainLoss, validLoss, mode):
    plt.figure(fig)
    plt.plot(trainLoss)
    plt.plot(validLoss)
    plt.title('Loss over Epochs ' + str(mode))
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Error)')
    plt.legend(['Training','Validation'], loc = 'top right')
    plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '_ValidationLoss.png', dpi = 100)
    plt.close()

def plot_accuracies(fig, trainAcc, validAcc, mode):
    plt.figure(fig)
    plt.plot(trainAcc)
    plt.plot(validAcc)
    plt.title('Accuracies over Epochs ' + str(mode))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training','Validation'], loc = 'top right')
    plt.savefig(os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '_ValidationAccuracies.png', dpi = 100)
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
    savepath = os.path.split(os.getcwd())[0] + '/results/' + str(mode) + '_ROC.png'
    plt.savefig(savepath, dpi=100)
    plt.close()



def plot_confusion_matrix(cm, classes,
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
            np.arange(0,len(mean_loss)), loss_lower, loss_upper,
            alpha=.2, label=r'$\pm$ std.'
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
            np.arange(0,len(mean_loss)), loss_lower, loss_upper,
            alpha=.3, label=r'$\pm$ std.'
        )
        plt.title(" Loss over Epochs - All Approaches" , fontsize=20)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Loss', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="upper right", fontsize=14)

    return mean_loss, loss_upper, loss_lower

def csv_save(method, auc):
    ''' Save AUCs scores to a csv file '''

    cols = ['AUC'+str(i+1) for i in range(auc.shape[1])]
    logs = pd.DataFrame(auc, columns=cols)    
    pth_to_save =  os.path.split(os.getcwd())[0] + "/results/" + method +  "_aucs.csv"
    logs.to_csv(pth_to_save)

    print(logs)