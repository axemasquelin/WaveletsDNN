from PIL import Image
import numpy as np
import pandas as pd
import cv2 ##Error with this library. System couldn't find the library. Reinstalled openCV and fixed the issue.
import os
import glob
import time
import matplotlib.pyplot as plt
from collections import namedtuple
import csv
import re
import shutil

###DICON Image reading not included yet. Use Python Package for reading DICON Images.
d
def normalize(image, MIN_BOUND, MAX_BOUND):
    image = image - 32768
       
    unique, counts = np.unique(image, return_counts=True)

    temp = image
    temp[temp==unique[0]] = MIN_BOUND #unique[1]
    image =(temp - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) #(image-np.min(image)) / (np.max(image)-np.min(image)) # 
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def deprocess_image(x,MIN_BOUND, MAX_BOUND):

    x= normalize(x, MIN_BOUND, MAX_BOUND)
 # normalize tensor: center on 0., ensure std is 0.1
    x = x - x.mean()
   # x /= (x.std() + 1e-5)
   # x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to grayscale array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x  


script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
record_location = os.path.split(script_path)[0]  #i.e. /path/to/dir
data_location = 'D:/med image/NIH/Images_png.zip/Images_png'
result_location = 'D:/med image/NIH/Images_png.zip/Results/Grayscale'
if (os.path.isdir(result_location) != True):
    print("Creating Result Directory")
    os.mkdir(result_location)
##Creating Destintation Folder for Images
des_location = result_location + '/new'
if (os.path.isdir(des_location) != True):
    print("Creating New Folder")
    os.mkdir(des_location)
##Creating Patch Folder for Patch Images
patch_location = result_location + '/patch'
if (os.path.isdir(patch_location) != True):
    print("Creating Patch Folder")
    os.mkdir(patch_location)
##Creating Comparison Folder for Images
comparison_location = result_location + '/compare'
if (os.path.isdir(comparison_location) != True):
    print("Creating Comparison Folder")
    os.mkdir(comparison_location)


with open(record_location + '/DL_info.csv') as f: 
    reader = csv.reader(f)
    next(reader) # skip header
    Coarse_lesion_types = []
    file_names = []
    coordinates = []
    DICOM_windows = []
    patient_ids = []
    study_ids = []
    key_ids = []
    lesion_diameters = []
    slice_ranges = []
    
    for row in reader:
        #image name is 1st column in the cvs file, lesion type is 10th, bounding box is 7th, 15th is the DICOM window value
        Coarse_lesion_type = row[9]
        file_name = row[0]
        coordinate = row[6]
        DICOM_window = row[14]
        patient_id = row[1]
        study_id = row[2]
        key_id = row[4]
        lesion_diameter = row[7]
        slice_range = row[11]

        ##generate the list from the cvs file for the lesion type and image names.
        Coarse_lesion_types.append(Coarse_lesion_type)
        file_names.append(file_name) 
        coordinates.append(coordinate)           
        DICOM_windows.append(DICOM_window)
        patient_ids.append(patient_id)
        study_ids.append(study_id)
        key_ids.append(key_id)
        lesion_diameters.append(lesion_diameter)
        slice_ranges.append(slice_range)

##convert the list to numpy array for operations
Coarse_lesion_types = np.asarray(Coarse_lesion_types, dtype=np.int16)
file_names = np.asarray(file_names)
coordinates = np.asarray(coordinates)
DICOM_windows = np.asarray(DICOM_windows)
patient_ids = np.asarray(patient_ids, dtype=int)
study_ids = np.asarray(study_ids, dtype=int)
key_ids = np.asarray(key_ids, dtype=int)
lesion_diameters = np.asarray(lesion_diameters)
slice_ranges = np.asarray(slice_ranges)

# check duplicate file name and remove the items from the arrays
unique, index, counts = np.unique(file_names, return_counts=True, return_index=True)
duplicate=unique[counts>1]
#print(duplicate)

Coarse_lesion_types = Coarse_lesion_types[index]
file_names = file_names[index]
coordinates = coordinates[index]
DICOM_windows = DICOM_windows[index]
patient_ids = patient_ids[index]
study_ids = study_ids[index]
key_ids = key_ids[index]
lesion_diameters = lesion_diameters[index]
slice_ranges = slice_ranges[index]

##find all the index of the lesion type is 5 (lung), then based on the index find the image name
file_names = file_names[np.where(Coarse_lesion_types == 5)[0]]
coordinates = coordinates[np.where(Coarse_lesion_types == 5)[0]]
DICOM_windows = DICOM_windows[np.where(Coarse_lesion_types == 5)[0]]
patient_ids = patient_ids[np.where(Coarse_lesion_types == 5)[0]]
study_ids = study_ids[np.where(Coarse_lesion_types == 5)[0]]
key_ids = key_ids[np.where(Coarse_lesion_types == 5)[0]]
lesion_diameters = lesion_diameters[np.where(Coarse_lesion_types == 5)[0]]
slice_ranges = slice_ranges[np.where(Coarse_lesion_types == 5)[0]]

    ##May want to introduce logic to compare slice ranges between two images of the same patient before comparing the data. --Axel
    ##This will negate, invalide comparision from different locations. --Axel

##Introduce comparison here? --Axel


dirlist = os.listdir(data_location) ##base dir

record = []
index = 0
patch_width = 72
patch_height = 72
NewImage_Count = 0

for dirname in dirlist: ##sub folder
    filelist = os.listdir(data_location + '/' + dirname)

    for filename in filelist: ##file name in the folder
        full_name = dirname + '_' + filename ##generate full image name matching the cvs file based on the subfolder and file name

        ## if file name exists in the cvs record, convert the HU value to the jpg format
        if full_name in file_names[:]:
            CompareImageCheck = True
            record = np.append(record, full_name)
            
            path = os.path.join(data_location, dirname)
            im_frame = Image.open(path + '/' + filename)
            np_frame = np.array(im_frame)
            col,row =  im_frame.size
            
            coordinate = np.asarray(re.findall(r"[-+]?\d*\.\d+|\d+", coordinates[index]), dtype=float)
            ##HU parameter value for normalization. Check Wiki or paper for suggest value for different tissue and organ
            MIN_BOUND,  MAX_BOUND = np.asarray(re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", DICOM_windows[index]), dtype=float)
            
            coordinate = np.round(coordinate)
            x_center = int((coordinate[2] + coordinate[0])/2)
            y_center = int((coordinate[3] + coordinate[1])/2)

            # lesion_diameter = np.asarray(re.findall(r"[-+]?\d*\.\d+|\d+", lesion_diameters[index]))
            # X_diameter = lesion_diameter[0]
            # Y_diameter = lesion_diameter[1]
           

            '''
            #plt.imshow(np_frame.reshape((col,row)))
            #plt.gray()
            #plt.show()
          
            plt.hist([ np_frame.flatten()], bins= 80, label=['original'], color='c')
            plt.legend(loc='upper right')
            plt.xlabel("Hounsfield Units (HU)")
            plt.ylabel("Frequency")
            plt.show()
            '''
    
            out_frame = deprocess_image(np_frame, MIN_BOUND, MAX_BOUND) 
            '''
            bins = int(np.max(out_frame)-np.min(out_frame) + 1)
            plt.hist([ out_frame.flatten()], bins = 'auto', label=['good y', 'converted'])
            plt.legend(loc='upper right')
            plt.show()
            '''
            out_frame = out_frame.reshape((col,row))
            #plt.imshow(out_frame)
            #plt.gray()
            #plt.show()

            patch = out_frame[(y_center - patch_height//2):(y_center + patch_height//2), (x_center - patch_width//2):(x_center + patch_width//2) ]
            ##Y is first just like in the excel bounding box (y-min, x-min, y-max, x-max)
            #New Images
            cv2.imwrite(os.path.join(des_location , full_name), np.uint8(out_frame)) ##(cv2.imwrite does not create the new folder if it doesn't exist)
            NewImage_Count = NewImage_Count + 1
            #Patch Images
            cv2.imwrite(os.path.join(patch_location , full_name), np.uint8(patch))

            index = index + 1

# Comparison Images
ComparisonFileName = []
ComparisonPatientID = []
ComparisonStudyID = []
ComparisonKeyID = []
ComparisonSliceRange = []

unique_pID, pID_index, pID_counts = np.unique(patient_ids, return_counts=True, return_index=True)
duplicate_ID = unique_pID[pID_counts>1]
duplicate_ID_counts = pID_counts[pID_counts>1]
duplicate_ID_index = pID_index[pID_counts>1]

index = 0
ComparisonImage_Count = 0

for ids in duplicate_ID:
    start_index = duplicate_ID_index[index]
    count = duplicate_ID_counts[index]
    study = study_ids[start_index: start_index + count] 
    study = np.asarray(study, dtype= int)
    key = key_ids[start_index: start_index + count]
    key = np.asarray(key, dtype = int)
    for i in range(duplicate_ID_counts[index]):
        for j in range(len(key)):
            if ((study[i] != study[j]) and ((key[j] <= key[i] + 15) and (key[j] >= key[i]-15))): ##Move Key Condition with study id condition. Introduce second for loop before the conditions. 
                #Creating Patient Specific Folder
                Image_name = "Patient_" + str(patient_ids[i + start_index]) + "_Key" + str(key_ids[i + start_index]) + "_Study_" + str(study_ids[i + start_index]) + '.png'
                patient_id = "Patient " + str(patient_ids[i + start_index])
                PatientFolder = comparison_location + "/" + patient_id
                if (os.path.isdir(PatientFolder) != True):
                    #print("Creating" + PatientFolder)
                    os.mkdir(PatientFolder)
                
                #Copying and Renaming Image to new convention (Patient ID_KeySlice_StudyID.png)
                file_name = file_names[i + start_index]
                shutil.copy(des_location + '/' + file_name, PatientFolder) #Copying Image to Compare Folder
                os.rename(PatientFolder + '/' + file_name, PatientFolder + '/' + Image_name)

                #Appending Lists that will be copied to a CSV File        
                ComparisonFileName.append(file_names[i + start_index])
                ComparisonPatientID.append(patient_ids[i + start_index])
                ComparisonStudyID.append(study_ids[i + start_index])
                ComparisonKeyID.append(key_ids[i + start_index])
                ComparisonSliceRange.append(slice_ranges[i + start_index])

                #Updating Comparison Image Count
                ComparisonImage_Count = ComparisonImage_Count + 1
                break
           
#                if((np.any(key[j] == np.arange(key[i]-15,key[i])) or (np.any(key[j] == np.arange(key[i]+1,key[i]+15))))):
 
    index = index + 1


# # CSV File Write
Comparison = pd.DataFrame()
Comparison['File Name'] = ComparisonFileName
Comparison['Patient ID'] = ComparisonPatientID
Comparison['Study ID'] = ComparisonStudyID
Comparison['Key Slice'] = ComparisonKeyID
Comparison['Slice Range'] = ComparisonSliceRange
CSV_FileName = result_location + '/ComparisonList.csv'
Comparison.to_csv(CSV_FileName, index=False)

# unique, counts = np.unique(file_names, return_counts=True)
print('finished')  
print('Number of New Images', NewImage_Count,  ' | Number of Comparison Images = ', ComparisonImage_Count)
print('-------------------------------------')