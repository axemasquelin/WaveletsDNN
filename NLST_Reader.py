import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd
import numpy as np
import types
import time
import glob
import pywt
import csv
import cv2
import os


'''
Project: 
Author: Axel Masquelin
Function Definition: Extract .nrrd CT images and metadata from the CT images.
'''
'''Load .nrrd Files and convert CT image information into array format'''
def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(itkimage.GetOrigin()))
    numpySpacing = np.array(list(itkimage.GetSpacing()))

    return numpyImage, numpyOrigin, numpySpacing

'''Take nrrd array and conduct wavelet transform'''
def wavelet(npImages, mode = 'HAAR'):
    #Compute Transform Coefficients
    coeffs = pywt.wavedec2(npImages, mode)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0.5

    #Wavelet Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H = imArray_H * 255
    imArray_H = np.uint8(imArray_H)
    return(imArray_H)

'''Take the raw CT image values and apply the Otsu Algorithm '''
def OtsuAlgorithm (npImages):
    _, OtsuValues = cv2.threshold(npImages,75,255,cv2.THRESH_BINARY)

    return OtsuValues

'''Read patient information from csv file (No patient information for dataset)'''
def readCSV(filename):
    lines = []
    with open(filename, "rb") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

'''Normalize CT image to grayscale for visual inspection'''
def normalizePlanes(npzarray):
    maxHU = np.max(npzarray) 
    minHU = np.min(npzarray)
    
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1
    npzarray[npzarray<0] = 0
    return npzarray

'''Generating CSV file of segmented patient information'''
# def csv_Writer(ImageNames, patientIDs, StudyIDs, lesionTypes):
#     patientData = pd.DataFrame()
#     patientData['filename'] = ImageNames
#     patientData['Patient ID'] = patientIDs
#     patientData['Study ID'] = StudyIDs
#     patientData['lesion Types'] = lesionTypes
#     #patientData['Lung Region'] = CTlocations
#     CSV_FileName = r"/media/lab/Linux/Medical Images/Boston Data/PatientInformation.csv"
#     patientData.to_csv(CSV_FileName, index=False)


########################################
#             Main Function            #
########################################'''

'''Defining/Creating Image SavePath'''
SavePath = r"/media/lab/Linux/Medical Images/Boston Data/Patient Images"
if (os.path.isdir(SavePath) != True):
    print("Creating Patient Image Directory")
    os.mkdir(SavePath)

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
    #CTlocation = splitResult[5]
    
    #print(patientID)
    #print(filename)
    '''Defining Patient Folders for SavePath'''
    PatientDirectory = SavePath + r'/' +  r'Patient ' + patientID
    WaveletDirectory = PatientDirectory + r'/Full_Reconstruction'
    GrayscaleDirectory = PatientDirectory + r'/Grayscale'
    if (os.path.isdir(PatientDirectory) != True):
        os.mkdir(PatientDirectory)
    if (os.path.isdir(WaveletDirectory) != True):
        os.mkdir(WaveletDirectory)
    if (os.path.isdir(GrayscaleDirectory) != True):
        os.mkdir(GrayscaleDirectory)
    
    WaveDest = WaveletDirectory + '/' + filename + '.jpg'
    CTDest = GrayscaleDirectory + '/' + filename + '.tiff'
    try: 
        '''Load Images'''
        numpyImage, numpyOrigin, numpySpacing = load_itk_image(image)
        numpyDimension = len(numpyImage.shape)
        print("Dimension: ", numpyDimension)
        print("Image Size: ", numpyImage.shape)
        print("Image Origin: ", numpyOrigin)
        print("Image Spacing: ", numpySpacing)

        # if (numpyDimension < 3):
            # '''Change Grayscale of Image'''
            # numpyImage= np.float32(normalizePlanes(numpyImage))
            # numpyImage = cv2.resize(numpyImage, (256, 256))  

            # '''Generating Otsu & Wavelet Image'''
            # waveImg = wavelet(numpyImage, 'db1')
            # grayImage = numpyImage * 255
            # OtsuImg = OtsuAlgorithm(grayImage) 
            # '''This solution is improper, as it does not properly conver the image
            # Find the Format of the image and conver https://stackoverflow.com/questions/10571874/opencv-imwrite-saving-complete-black-jpeg'''
            # '''Save Image'''
            # #cv2.imwrite(CTDest, grayImage)
            # cv2.imshow("Image", numpyImage)
            # #cv2.imshow("Otsu Image", OtsuImg)
            # #cv2.imshow("Wavelet Imagee", waveImg)
            # key = cv2.waitKey(0)
            # if (key == 1048603): #Escape Key
            #     cv2.destroyAllWindows
            #     break
            # cv2.destroyAllWindows
            # '''Wavelet Deconstruction & Reconstruction'''
            # # waveImg = wavelet(numpyImage, 'db1')
            # # cv2.imwrite(WaveDest, waveImg)

            #print('\n')
        if (numpyDimension >= 3): #else:
            #for z in range (len(numpyImage[:,:,1])): 
            x = 0
            y = 0
            z = 0
            
            while ((x <= len(numpyImage[1,:,:])) or (y <= len(numpyImage[1,:,:])) or (z <= len(numpyImage[1,:,:]))):
                if ((z == 0) and (y == 0) and (x ==0)):
                    slices = numpyImage[:,:,z]   
                slices = np.float32(normalizePlanes(slices))
                # '''Save Image'''
                #grayImage = numpyImage * 255  
                # '''This solution is improper, as it does not properly conver the image
                # Find the Format of the image and conver https://stackoverflow.com/questions/10571874/opencv-imwrite-saving-complete-black-jpeg'''
                # #cv2.imwrite(CTDest, grayImage)
                slices = cv2.resize(slices, (256, 256))
                waveImg = wavelet(slices, 'db1')
                OtsuImg = OtsuAlgorithm((slices*255))
                #cv2.imshow('Otsu', OtsuImg)
                cv2.imshow('wavelet', waveImg)
                cv2.imshow('gray CT', slices)
                key = cv2.waitKey(0)
                # '''Wavelet Deconstruction & Reconstruction'''
                # # waveImg = wavelet(numpyImage, 'db1')
                # # cv2.imwrite(WaveDest, waveImg)
                #print('\n')
                '''Key Input to Move Throught 3D CT Image'''
                if ((key == 1113941) and (z != 63)): #Page UP 
                    cv2.destroyAllWindows
                    z = z + 1
                    slices = numpyImage[:,:,z] 
                if ((key == 1113942) and (z != 0)): #Page Down
                    cv2.destroyAllWindows
                    z = z - 1
                    slices = numpyImage[:,:,z] 
                if ((key == 1113938) and (y != 63)): #Up Key
                    cv2.destroyAllWindows
                    y = y + 1
                    slices = numpyImage[:,y,:] 
                if ((key == 1113940) and (y != 0)): #Down Key
                    cv2.destroyAllWindows
                    y = y - 1
                    slices = numpyImage[:,y,:]  
                if ((key == 1113937) and (x != 0)): #Left Key
                    cv2.destroyAllWindows    
                    x = x - 1
                    slices = numpyImage[x,:,:] 
                if ((key == 1113939) and (x !=63)): #Right Key
                    cv2.destroyAllWindows
                    x = x + 1
                    slices = numpyImage[x,:,:] 
                if key == 1048603: #Escape
                    cv2.destroyAllWindows 
                    break
                

 


    except Exception as err:
        print('Error: ', err)
        ErrorList.append(image)

        #print('\n')
    
#     patientIDs.append(patientID)
#     StudyIDs.append(StudyID)
#     lesionTypes.append(lesionType)
#     #CTlocations.append(CTlocation)

# csv_Writer(ImageNames, patientIDs, StudyIDs, lesionTypes)