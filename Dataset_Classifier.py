from wavelets_V2 import WaveCoeff
import SimpleITK as sitk
import numpy as np
import types
import time
import glob
import pywt
import csv
import cv2
import os

'''
Main Function
Author: Axel Masquelin
Definition: Purpose of the function is to generate dataset from 3D ct image at nodule
maximum size and utilize patient csv information to classify as cancerous or benign. 
'''

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(itkimage.GetOrigin()))
    numpySpacing = np.array(list(itkimage.GetSpacing()))

    return numpyImage, numpyOrigin, numpySpacing

def normalizePlanes(npzarray):
    maxHU = np.max(npzarray) 
    minHU = np.min(npzarray)
    
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1
    npzarray[npzarray<0] = 0
    return npzarray

def csvReader(patientID):
    records = r"/media/lab/Linux/Medical Images/Boston Data/PatientCancerList_082918.csv"
    classification = []
    groupID = []
    with open(records) as f:
        reader = csv.reader(f)
        next(reader) #Skips Headers
        for row in reader:
            if patientID == row[0]:
                classification = row[1] #Classes: (0 - Noncancerous , 1 - cancerous)
                groupID = row[2]
    return classification, groupID


'''Take nrrd array and conduct wavelet transform'''
def wavelet(npImages, mode):   
    initflag = 1
    imgScale = 0
    coeffs = pywt.wavedec2(npImages, mode)
    for n in range(len(coeffs)-1):
    
        if initflag == 0:
            temp1 = np.concatenate((imgScale, coeffs[n+1][0]), axis = 1)
            temp2 = np.concatenate((coeffs[n+1][1], coeffs[n+1][2]), axis = 1)
            imgScale = np.concatenate((temp1, temp2), axis = 0)
                
        else:
            temp1 = np.concatenate((coeffs[n], coeffs[n+1][0]), axis = 1)
            temp2 = np.concatenate((coeffs[n+1][1], coeffs[n+1][2]), axis = 1)
            imgScale = np.concatenate((temp1, temp2), axis = 0)
            initflag = 0
    
    cv2.imshow('Image', imgScale)
    return(imgScale*10)

def CT_Slicer(numpyImage, patientID, studyID, lesionType):
    n = 32
    WaveG0_Noncancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Wavelet/db1/Group0/Noncancerous"
    WaveG0_Cancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Wavelet/db1/Group0/Cancerous"
    WaveG1_Noncancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Wavelet/db1/Group1/Noncancerous"
    WaveG1_Cancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Wavelet/db1/Group1/Cancerous"
    if (os.path.isdir(WaveG0_Noncancerous) != True):
        print("Creating WaveG0 Noncancerous Directory")
        os.mkdir(WaveG0_Noncancerous)
    if (os.path.isdir(WaveG0_Cancerous) != True):
        print("Creating WaveG0 Cancerous Directory")
        os.mkdir(WaveG0_Cancerous)
    if (os.path.isdir(WaveG1_Noncancerous) != True):
        print("Creating WaveG1 Noncancerous Directory")
        os.mkdir(WaveG1_Noncancerous)
    if (os.path.isdir(WaveG1_Cancerous) != True):
        print("Creating WaveG1 Cancerous Directory")
        os.mkdir(WaveG0_Cancerous)    

    G0_Noncancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group0/Noncancerous"
    G0_Cancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group0/Cancerous"
    G1_Noncancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group1/Noncancerous"
    G1_Cancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT/Grayscale/Group1/Cancerous"
    if (os.path.isdir(G0_Noncancerous) != True):
        print("Creating Patient Image Directory")
        os.mkdir(G0_Noncancerous)
    if (os.path.isdir(G0_Cancerous) != True):
        print("Creating Patient Image Directory")
        os.mkdir(G0_Cancerous)
    if (os.path.isdir(G1_Noncancerous) != True):
        print("Creating Patient Image Directory")
        os.mkdir(G1_Noncancerous)
    if (os.path.isdir(G1_Cancerous) != True):
        print("Creating Patient Image Directory")
        os.mkdir(G1_Cancerous)
    
    #Getting X, Y, Z slices of ROI
    Xslice = numpyImage[n,:,:]
    Yslice = numpyImage[:,n,:]
    Zslice = numpyImage[:,:,n] 
    
    #Converting Image to Grayscale
    Xslice = np.float32(normalizePlanes(Xslice))
    Yslice = np.float32(normalizePlanes(Yslice))
    Zslice = np.float32(normalizePlanes(Zslice))


    # cv2.imshow('Image', Xwavelet)
    # cv2.waitKey(0)

    #Rescaling Images from 64x64 to 256x256
    Xslice = cv2.resize(Xslice, (256, 256))
    Yslice = cv2.resize(Yslice, (256, 256))
    Zslice = cv2.resize(Zslice, (256, 256))
    
    #Converting Image to Recompose Wavelet
    Xwavelet = wavelet(Xslice, 'db1')*255
    Ywavelet = wavelet(Yslice, 'db1')*255
    Zwavelet = wavelet(Zslice, 'db1')*255

    Xslice = Xslice * 255    
    Yslice = Yslice * 255
    Zslice = Zslice * 255  
    

    # '''Save Image'''
    classification, groupID = csvReader(patientID)
    if (groupID == str(0)):
        if classification  == str(0):
            xWaveName = WaveG0_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'xWaveSlice' + '_' + str(classification) + '.jpg'
            yWaveName = WaveG0_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'yWaveSlice' + '_' + str(classification) + '.jpg'
            zWaveName = WaveG0_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'zWaveSlice' + '_' + str(classification) + '.jpg'
            xSliceName = G0_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'xSlice' + '_' + str(classification) + '.jpg'
            ySliceName = G0_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'ySlice' + '_' + str(classification) + '.jpg'
            zSliceName = G0_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'zSlice' + '_' + str(classification) + '.jpg'
            # cv2.imwrite(xWaveName, Xwavelet)
            # cv2.imwrite(yWaveName, Ywavelet)
            # cv2.imwrite(zWaveName, Zwavelet)
            # cv2.imwrite(xSliceName, Xslice)
            # cv2.imwrite(ySliceName, Yslice)
            # cv2.imwrite(zSliceName, Zslice)
        else:
            xWaveName = WaveG0_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'xWaveSlice' + '_' + str(classification) + '.jpg'
            yWaveName = WaveG0_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'yWaveSlice' + '_' + str(classification) + '.jpg'
            zWaveName = WaveG0_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'zWaveSlice' + '_' + str(classification) + '.jpg'
            xSliceName = G0_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'xSlice' + '_' + str(classification) + '.jpg'
            ySliceName = G0_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'ySlice' + '_' + str(classification) + '.jpg'
            zSliceName = G0_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'zSlice' + '_' + str(classification) + '.jpg'
            # cv2.imwrite(xWaveName, Xwavelet)
            # cv2.imwrite(yWaveName, Ywavelet)
            # cv2.imwrite(zWaveName, Zwavelet)
            # cv2.imwrite(xSliceName, Xslice)
            # cv2.imwrite(ySliceName, Yslice)
            # cv2.imwrite(zSliceName, Zslice)

    else:
        if classification == str(0):
            xWaveName = WaveG1_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'xWaveSlice' + '_' + str(classification) + '.jpg'
            yWaveName = WaveG1_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'yWaveSlice' + '_' + str(classification) + '.jpg'
            zWaveName = WaveG1_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'zWaveSlice' + '_' + str(classification) + '.jpg'
            xSliceName = G1_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'xSlice' + '_' + str(classification) + '.jpg'
            ySliceName = G1_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'ySlice' + '_' + str(classification) + '.jpg'
            zSliceName = G1_Noncancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'zSlice' + '_' + str(classification) + '.jpg'
            # cv2.imwrite(xWaveName, Xwavelet)
            # cv2.imwrite(yWaveName, Ywavelet)
            # cv2.imwrite(zWaveName, Zwavelet)
            # cv2.imwrite(xSliceName, Xslice)
            # cv2.imwrite(ySliceName, Yslice)
            # cv2.imwrite(zSliceName, Zslice)
        else:
            xWaveName = WaveG1_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'xWaveSlice' + '_' + str(classification) + '.jpg'
            yWaveName = WaveG1_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'yWaveSlice' + '_' + str(classification) + '.jpg'
            zWaveName = WaveG1_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'zWaveSlice' + '_' + str(classification) + '.jpg'
            xSliceName = G1_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'xSlice' + '_' + str(classification) + '.jpg'
            ySliceName = G1_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'ySlice' + '_' + str(classification) + '.jpg'
            zSliceName = G1_Cancerous + '/' + str(patientID) + '_' + str(studyID) + '_' + str(lesionType) + '_' + 'zSlice' + '_' + str(classification) + '.jpg'
            # cv2.imwrite(xWaveName, Xwavelet)
            # cv2.imwrite(yWaveName, Ywavelet)
            # cv2.imwrite(zWaveName, Zwavelet)
            # cv2.imwrite(xSliceName, Xslice)
            # cv2.imwrite(ySliceName, Yslice)
            # cv2.imwrite(zSliceName, Zslice)
            

    return()
    
###############################
#       Main Function         #
###############################
'''Defining/Creating Image SavePath'''
WholeChestPath = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Whole CT"
if (os.path.isdir(WholeChestPath) != True):
    os.mkdir(WholeChestPath)

PatchPath = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Patch CT"
if (os.path.isdir(PatchPath) != True):
    os.mkdir(PatchPath)

GrayscalePath = WholeChestPath + '/Grayscale'
if (os.path.isdir(GrayscalePath) != True):
    os.mkdir(GrayscalePath)

GrayscalePath = PatchPath + '/Grayscale'
if (os.path.isdir(GrayscalePath) != True):
    os.mkdir(GrayscalePath)

'''NRRD Image Locations'''
nrrdFilePath = r"/media/lab/Linux/Medical Images/Boston Data/NLST-NoduleData"
ImageNames = glob.glob(nrrdFilePath + r'/*.nrrd')
'''Generating Empty List to be saved into CSV File'''
patientIDs = []
StudyIDs = []
lesionTypes = []
CTlocations = [] 
ErrorList = []

for image in ImageNames:
    filename = str(os.path.basename(image))
    filename = os.path.splitext(filename)[0]
    splitResult = filename.split("_")
    
    patientID = splitResult[0]
    StudyID = splitResult[1]
    lesionType = splitResult[2]

    try:
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(image)
        numpyDimension = len(numpyImage.shape)
        # print("Dimension: ", numpyDimension)
        # print("Image Size: ", numpyImage.shape)
        # print("Image Origin: ", numpyOrigin)
        # print("Image Spacing: ", numpySpacing)

        if(numpyDimension == 3):
            CT_Slicer(numpyImage, patientID, StudyID, lesionType)
        
        else:
            
            WholeCT_G0_Noncancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Whole CT/Grayscale/Group0/Noncancerous"
            WholeCT_G0_Cancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Whole CT/Grayscale/Group0/Cancerous"
            WholeCT_G1_Noncancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Whole CT/Grayscale/Group1/Noncancerous"
            WholeCT_G1_Cancerous = r"/media/lab/Linux/Medical Images/Boston Data/DataSet/Whole CT/Grayscale/Group1/Cancerous"
            if (os.path.isdir(WholeCT_G0_Noncancerous) != True):
                print("Creating Whole CT Group 0 Directory")
                os.mkdir(WholeCT_G0_Noncancerous)
            if (os.path.isdir(WholeCT_G0_Cancerous) != True):
                print("Creating Whole CT Group 0 Directory")
                os.mkdir(WholeCT_G0_Cancerous)
            if (os.path.isdir(WholeCT_G1_Noncancerous) != True):
                print("Creating Whole CT Group 0 Directory")
                os.mkdir(WholeCT_G1_Noncancerous)
            if (os.path.isdir(WholeCT_G1_Cancerous) != True):
                print("Creating Whole CT Group 1 Directory")
                os.mkdir(WholeCT_G1_Cancerous)
            
            '''Change Grayscale of Image'''
            numpyImage= np.float32(normalizePlanes(numpyImage))
            numpyImage = cv2.resize(numpyImage, (256, 256))
            rgbslices = cv2.cvtColor(numpyImage, cv2.COLOR_GRAY2RGB)  
            #coeffs = WaveCoeff(numpyImage, 'db1')
            Wavelet = wavelet(numpyImage, 'db1')
            '''Generating Otsu & Wavelet Image'''
            grayImage = numpyImage * 255

            '''Save Image'''
            classification, groupID = csvReader(patientID)
            # if (groupID == str(0)):
            #     '''Group 00'''
            #     if classification == str(0):
            #         ImgName = WholeCT_G0_Noncancerous + '/' + filename + '.jpg'
            #         cv2.imwrite(ImgName, grayImage)
            #     else:
            #         ImgName = WholeCT_G0_Cancerous + '/' + filename + '.jpg'
            #         cv2.imwrite(ImgName, grayImage)
            # else:
            #     '''Group 01'''
            #     if classification  == str(0):
            #         ImgName = WholeCT_G1_Noncancerous + '/' + filename + '.jpg'
            #         cv2.imwrite(ImgName, grayImage)
            #     else:
            #         ImgName = WholeCT_G1_Cancerous + '/' + filename + '.jpg'
            #         cv2.imwrite(ImgName, grayImage)

    except Exception as err:
        print('Error: ', err)
        ErrorList.append(image)