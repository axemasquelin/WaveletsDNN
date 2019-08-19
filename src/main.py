# coding: utf-8

'''
    Project: Lung Cancer Wavelet Neural Network
    Date: 19/12/2018
    Axel Masquelin
'''

# libraries and dependencies
# --------------------------------------------
from sklearn.metrics import roc_curve, auc, confusion_matrix
from preprocessing import dwcoeff, dwdec
from train_test import *
from models import *
from utils import *

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.tensor
import torchvision
import torch

import matplotlib.pyplot as plt
import preprocessing
import numpy as np
import cv2, sys, utils, os
# --------------------------------------------

def progressBar(value, endvalue, bar_length=50):
    percent = float(value) / endvalue
    arrow = '=' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r[{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    print('\n')

if __name__ == '__main__':  
    # #Variable Definition
    modes = ['Inception']
    modes1 = ['WaveCon','WaveS', 'Raw', 'AlexNet'] #'Raw', 'AlexNet', 'Conv3', 'Alex1', 'Alex2', 'Alex3',  'Alex4', 'AlexNet']
    modes2 = ['AlexNet']
    class_names = ['benign','malignant']
    check_grad = True
    reps = 2
    epch = 250
    seed = 2019
    fig = 2
    #GPU Init"""  """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    LossEpoch = np.zeros((len(modes),1))
    for i in range(len(modes)):
        #Progress Bar
        progressBar(i+1, len(modes))
        
        
        #Dataset
        # if ((modes[i] == 'WaveS')): #Wavelet Scalogram
        #     TrainPath = r'/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Wavelet/db1/Group1/'
        #     TestPath = r'/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Wavelet/db1/Group0/'

        # else: #All Other Dataset
        TrainPath = r'/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group1/'
        TestPath = r'/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group0/'    
        
        #Network Init & Weight Init
        print("Testing Mode: " + str(modes[i]))
        if ((modes[i] == 'Raw') or (modes[i] == 'WaveD') or (modes[i] == 'WaveS')):
            transform = transforms.Compose([transforms.Resize(128, interpolation = 2),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Grayscale(),
                                            transforms.ToTensor()])
            net = NoConv_256()
            net.apply(init_weights)
            
        elif (modes[i] == 'WaveCon'):
            transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
            net = NoConv_4D()
            net.apply(init_weights)

        elif (modes[i] == 'Conv3'):
            feat = 3
            transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
            net = Conv_3Fil()
            net.apply(init_weights)

        elif (modes[i] == 'Conv98'):
            feat = 3
            transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
            net = Conv_98Fil()
            net.apply(init_weights)

        elif (modes[i] == 'Alex1'):
            transform = transforms.Compose([transforms.Resize(224, interpolation = 2), transforms.ToTensor()])
            net = alexnet_conv1()
            print(net)
            set_parameter_requires_grad(net, feature_extracting= False)
            
            # for name, param in net.named_parameters():
            #     if check_grad == True:
            #         print(name, param)

        elif (modes[i] == 'Alex2'):
            transform = transforms.Compose([transforms.ToTensor()])
            net = alexnet_conv2()
            print(net)
            set_parameter_requires_grad(net, feature_extracting= False)
            
            # for name, param in net.named_parameters():
            #     if check_grad == True:
            #         print(name, param)
        
        elif (modes[i] == 'Alex3'):
            transform = transforms.Compose([transforms.ToTensor()])
            net = alexnet_conv3()
            print(net)
            set_parameter_requires_grad(net, feature_extracting= False)
            
            # for name, param in net.named_parameters():
            #     if check_grad == True:
            #         print(name, param)    
        
        elif (modes[i] == 'Alex4'):
            transform = transforms.Compose([transforms.ToTensor()])
            net = alexnet_conv4()
            print(net)
            set_parameter_requires_grad(net, feature_extracting= False)
            
            # for name, param in net.named_parameters():
            #     if check_grad == True:
            #         print(name, param)
        
        elif (modes[i] == 'Inception'):
            size = 299
            transform = transforms.Compose([transforms.Resize(size, interpolation = 2),
                            # transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                            transforms.RandomHorizontalFlip(p=0.8),
                            # transforms.TenCrop(size, vertical_flip=False),
                            transforms.ToTensor()
                           ])
            net = torchvision.models.inception_v3(pretrained = True)
            print(net)
            set_parameter_requires_grad(net,feature_extracting = True)
            net.fc = nn.Linear(2048,2)
            # net.classifier[4] = nn.Linear(50, 10)
            # net.classifier[6] = nn.Linear(10, 2)
            print(net)

        #Sending Net to GPU
        net = net.to(device)

        static_Fig = 0
        mean_fig = 1
        fprs, tprs, loss = [], [], []
        auc_scores = np.zeros((1, reps))

        for r in range(reps):
            trainset = torchvision.datasets.ImageFolder(TrainPath, transform = transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size= 100, shuffle= True)
            testset = torchvision.datasets.ImageFolder(TestPath, transform = transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size= 100, shuffle= True)
            classes = (0,1) #('noncancerous', 'cancerous')

            trainLoss, validLoss, trainAcc, validAcc = train(trainloader, testloader, net, device, r, mode = modes[i], epochs = epch, lrs = 0.001, moment = 0.99)
            loss.append(trainLoss)
            plot_losses(fig, trainLoss, validLoss, modes[i])
            plot_accuracies(fig, trainAcc, validAcc, modes[i])
                
            confmatrix, fp, tp, fils, raw = test(testloader, net, device, mode = modes[i])
            
            fprs.append(fp), tprs.append(tp)

            net.apply(init_weights)
    
            class_names = ["Malignant", "Benign"]
            plt.figure(fig)
            plot_confusion_matrix(confmatrix, classes= class_names, normalize = True, title = 'Normalize Confusion Matrix ' + str(modes[i]))
            fig += 1
    
        #Get Conv Filters
        if ((modes[i] == 'Conv3')):
            #Raw Image
            cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + modes[i] + "_Original.png", raw[0,:,:]*255)
            preprocessing.pywtFilters(raw, 'db1')
            #Convolution Activation
            img = np.zeros((128,128,3))
            img[:,:,0] = fils[0,0,:,:]
            img[:,:,1] = fils[0,1,:,:]
            img[:,:,2] = fils[0,2,:,:]
            # img[:,:,3] = fils[0,3,:,:]
            # img = np.transpose(fils[0], (2,1,0))
            #Convert this into a loop
            cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + modes[i] + "_Allfilter.png", img*1000)
            cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + modes[i] + "_filter01.png", fils[0,0,:,:]*1000)
            cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + modes[i] + "_filter02.png", fils[0,1,:,:]*1000)
            cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + modes[i] + "_filter03.png", fils[0,2,:,:]*1000)
            # cv2.imwrite(os.path.split(os.getcwd())[0] + "/results/filters/" + modes[i] + "_filter03.png", fils[0,3,:,:]*1000)
            
        
        print('fprs: ' + str(fprs))
        print('tprs: ' + str(tprs))
        auc_scores[0,:] = calcAuc(fprs,tprs, modes[i], fig, plot_roc= True)
        fig += 1
        
        csv_save(modes[i], auc_scores)           
        
        mean_losses, loss_upper, loss_lower = calcLoss_stats(loss, modes[i], static_Fig, fig, plot_loss = True, plot_static= True)
        fig += 1

        print("Length of Mean: " + str(len(mean_losses)))
        print("Length of Upper: " + str(len(loss_upper)))
        print("Length of Lower: " + str(len(loss_lower)))

        plt.figure(mean_fig)
        plt.plot(mean_losses)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss (Mean Error)', fontsize=14)
        plt.title(" Average Loss over Epochs")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(modes, loc = 'top right')
    
    
        plt.fill_between(
            np.arange(0,epch), loss_lower, loss_upper,
            alpha=.2, label=r'$\pm$ std.'
            )
        savepath = os.path.split(os.getcwd())[0] + '/results/AllApproaches_AverageLoss.png'
        plt.savefig(savepath, dpi = 100)

    plt.show()