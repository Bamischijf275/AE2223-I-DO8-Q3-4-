# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:32:55 2022

@author: huege
"""
# INIT
# startup
import time

T00 = time.time()

# standard modules
import cv2 as cv
#import imutils
#import math
import numpy as np
#import numpy.random as rnd
import os
#import pandas as pd
import statistics as stat
import sys
import warnings
#from scipy import ndimage
#from skimage.feature import peak_local_max
#from skimage.segmentation import watershed
#import PIL

# own functions
from FUNCTIONS_V2 import *

# init
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)

# MAIN
print("\n----- START PROGRAM ----- \n")

# Macro parameters
Loop = "List"  # Range, Random, List, All
N,M,K = [],[],[50,1]
Name = "Tape_B"
Tape= "Cropped" #Large, Cropped, none=smalls

Detail = [["", "", "save"], 250]  # draw/print/save, substep Dt
TypeOUT = [".jpg", ".csv"]
Save = ["", "Matrix", ""] #"Img", "Matrix", "Extra"

Compute = ["","","CP"] #WT,CV,CP
CV=["CROP"] #CROP, TIFtoCSV

errorMin = 10**(-3)
o = 2 # decimals

# Program parameters
WT_Parameters = [2.5, [0.5, 5, 2], [5, 20, 2], 3, "exact","UF"]  # Radius, Relative errors, Filter, kernel
#WT_Parameters = [4.5, [0.4, 5, 2], [6, 48, 2], 3, "exact",""]  # <- Combined Score, Fiber ID ^

CP_Parameters = [0.8]  # cutoff

M = [0,1,2] #choose plots (area,fiber,own metrics)

# file names
WT_PathIN = "../Data/Tape_B/"
WT_PathOUT = "../Data Processed/Watershed/"
WT_Type = [".jpg", ".png", ".csv"]  # in, out_img, out_matrix
        
CV_PathIN = "../Data Processed/Watershed/"
CV_PathOUT = "../Data Processed/Watershed/" #.csv and Out-images

CP_PathIN = "../Data Processed/"
CP_GroundTruth = "Annotated/mask_csv"   #sub-folder
CP_Algorithms = [               #sub-folders
        #"Annotated",
        "AI results/dataset1/",
        "AI results/dataset2/",
        "AI results/dataset3/",
        "AI results/dataset4/",
        "Watershed"
        ]

CP_PathOUT = "../Data Processed/Comparator/" # for extras, ?excel?

# process
path_script = os.path.dirname(__file__)

# Waterhsed
if "WT" in Compute:
    print("\n WATERSHED : \n")
    
    if Tape == "Large" or Tape == "Cropped":
        WT_PathIN += "UncroppedImages/"
        WT_Type[1] = "L"+WT_Type[1]
        WT_Type[2] = "L"+WT_Type[2] # consistent naming scheme (Large pictures)
    else: 
        WT_PathIN += "Images/"
    
    Names = NAMES(Loop,N,M,K,Tape,Name)
    Names = list(dict.fromkeys(Names))
    print("Filenames :")
    for name in Names:
        print(name)
        
    Progress = len(Names)
    progress = 1
    for name in Names:
        print("\n\n -- NEWFILE : ", name, " - ",progress,"/",Progress," -- \n")
        T0 = time.time()
        
        # file
        FileName = name + WT_Type[0]
        path = os.path.join(path_script, WT_PathIN, FileName)
        print(path)
        WT_Image = cv.imread(path)
        print(path)

        WT = WATERSHED(WT_Image, WT_Parameters, Detail)
        WT_Image = WT[0]
        WT_Matrix = WT[1]
    
        if "Img" in Save:
            FileName = name + WT_Type[1]
            print("saving WT Image : ", FileName)
            path = os.path.join(path_script, WT_PathOUT)
            os.chdir(path)
            path = str(path + FileName)
            cv.imwrite(path, WT_Image)
    
        if "Matrix" in Save:
            FileName = name + WT_Type[2]
            print("saving WT Matrix : ", FileName)
            path = os.path.join(path_script, WT_PathOUT)
            os.chdir(path)
            path = str(path + FileName)
            #pd.DataFrame(WT_Matrix).to_csv((path), header="none", index="none")
            np.savetxt(path, WT_Matrix, delimiter=",")
    
        if "Extra" in Save:
            IMG = WT[2]
            i = 0
            for Img in IMG:
                img, step = Img
                FileName = name + "_step_" + str(i) + "_" + step + WT_Type[1]
                print("saving extra WT Image : ", FileName)
                path = os.path.join(path_script, WT_PathOUT)
                os.chdir(path)
                path = str(path + "Extras/" + FileName)
                cv.imwrite(path, img)
                i += 1
            
        progress += 1
        T1 = time.time()
        print("> " + str(round((T1 - T0), 1)) + "[s] <")

     # Converter
if "CV" in Compute:
    print("\n CONVERTER : \n")
    
    Names = NAMES(Loop,N,M,K,Tape,Name)
    # remove _2_ if "CROP"
    Names = list(dict.fromkeys(Names))
    print("Filenames :")
    for name in Names:
        print(name)
    
    Progress = len(Names)
    progress = 1
    for name in Names:
        print("\n\n -- NEWFILE : ", name, " - ",progress,"/",Progress," -- \n")
        T0 = time.time()
        
        # various converters
        if "TIFtoCSV" in CV:
            CV_Type=[".tif",".csv"]
        
            PathIN = os.path.join(path_script, CV_PathIN, name + CV_Type[0])
            PathOUT = os.path.join(path_script, CV_PathOUT, name + CV_Type[1])
            CONVERT_TIFtoCSV(PathIN, PathOUT)
            print("Converted",str(name+CV_Type[0]),"to",str(name+CV_Type[1]))
        
        if "CROP" in CV: #crop matrix   
            CV_Type=[".tif",".csv"]
            
            FileName = name + "L" + CV_Type[1]
            path = os.path.join(path_script, CV_PathIN, FileName)
            Arr = np.genfromtxt(path, delimiter=",")
            
            #Arr_crop = CONVERT_CROP(Arr, 2, 5)
            Arr_crop = CONVERT_CROP2(Arr)
            #print(len(Arr_crop))
            
            path = os.path.join(path_script, CV_PathOUT)
            os.chdir(path)
            print("Cropped : ", Arr.shape, "to", Arr_crop[0].shape)
            
            sep = '_'
            n = name.split(sep)[-1]
            m = 1
            for Arr in Arr_crop:
                FileName = Name + "_" + str(n) + "_" + str(m)
                print("saving :",FileName)
                path = os.path.join(path_script, CV_PathOUT, FileName + CV_Type[1])
                #pd.DataFrame(Arr).to_csv((path), header="none", index="none")
                np.savetxt(path, Arr, delimiter=",")
                m += 1
                
        progress += 1
        T1 = time.time()
        print("> " + str(round((T1 - T0), 1)) + "[s] <")
        
if Tape == "Cropped":
    Tape = ""

# Comparator
if "CP" in Compute:
    print("\n COMPARATOR : \n")
    
    Names = NAMES(Loop,N,M,K,Tape,Name)
    Names = list(dict.fromkeys(Names))
    print("Filenames :")
    for name in Names:
        print(name)
        
    CP_Confusion = []
    CP_res = []
    CP_stat = []
    a = 0
    while a < len(CP_Algorithms):
        CP_Confusion.append([[0,0,0,0],[0,0,0,0]]) # Fibers
        CP_res.append([
                    [[],[],[],[]], # area confusion
                    [[],[],[],[]], # fiber confusion
                    [[],[],[],[]] # metrics
                    ])
        a += 1
        
    Progress = len(Names)
    progress = 1
    for name in Names:
        print("\n\n -- NEWFILE : ", name, " - ",progress,"/",Progress," -- \n")
        T0 = time.time()

        CP_Type = [".csv", ".png"]

        # file
        path = os.path.join(path_script, CP_PathIN, CP_GroundTruth, name + CP_Type[0])
        #print(path)
        CP_MatrixT = np.genfromtxt(path, delimiter=",")
    
        # compare each algo for each file
        a=0
        for Alg in CP_Algorithms:
            # File
            print(Alg, "against", CP_GroundTruth)
            
            #if Tape == "Cropped":
                 #associate tape
                 #path = os.path.join(path_script, CP_PathIN, Alg, name + CP_Type[0])
            #else:
            path = os.path.join(path_script, CP_PathIN, Alg, name + CP_Type[0])
            CP_MatrixR = np.genfromtxt(path, delimiter=",")
            #print(path)
    
            # Comp
            CP = COMPARATOR(CP_MatrixT, CP_MatrixR, CP_Parameters, Detail)
            result = CP[0]
    
            # Results
            
            # Area
            CP_Confusion[a][0][0] += result[0][0]
            CP_Confusion[a][0][1] += result[0][1]
            CP_Confusion[a][0][2] += result[0][2]
            CP_Confusion[a][0][3] += result[0][3]
            # Fibers
            CP_Confusion[a][1][0] += result[1][0]
            CP_Confusion[a][1][1] += result[1][1]
            CP_Confusion[a][1][2] += result[1][2]
            CP_Confusion[a][1][3] += result[1][3]
            
            # Area
            CP_res[a][0][0].append(result[0][0]/result[0][3])
            CP_res[a][0][1].append(1-result[0][1]/result[0][3])
            CP_res[a][0][2].append(1-result[0][2]/result[0][3])
            CP_res[a][0][3].append(result[0][3])
            # Fibers
            CP_res[a][1][0].append(result[1][0]/result[1][3])
            CP_res[a][1][1].append(1-result[1][1]/result[1][3])
            CP_res[a][1][2].append(1-result[1][2]/result[1][3])
            CP_res[a][1][3].append(result[1][3])
            # Metrics
            CP_res[a][2][0].append(result[2][0])
            CP_res[a][2][1].append(1-result[2][1])
            CP_res[a][2][2].append(1-result[2][2])
            CP_res[a][2][3].append(1-result[2][3])

            # Save images
            if "Extra" in Save:
                IMG = CP[1]
                i = 0
                for Img in IMG:
                    img, step = Img
                    FileName = name + "_step_" + str(i) + "_" + step + CP_Type[1]
                    print("saving extra CP Image : ", FileName)
                    path = os.path.join(path_script, CP_PathOUT)
                    os.chdir(path)
                    path = str(path + FileName)
                    cv.imwrite(path, img)
                    i += 1
            a += 1
            
        progress += 1
        T1 = time.time()
        print("> " + str(round((T1 - T0), 1)) + "[s] <")

if "CP" in Compute:
    a = 0
    # Algo Stats
    print("\n CONCLUSION : \n")
    print("total True Area: ", sum(CP_res[a][0][3][:]))
    print("total number of True fibers : ", sum(CP_res[a][1][3][:]))
    
    while a < len(CP_Algorithms):
        print("\n Comparing ",CP_Algorithms[a], "against", CP_GroundTruth, ": ")
        print("Area")
        print("     TP:", round(stat.mean(CP_res[a][0][0])*100,o), "%")
        print("     FP:", round(stat.mean(CP_res[a][0][1])*100,o), "%")
        print("     FN:", round(stat.mean(CP_res[a][0][2])*100,o), "%")
        
        print("Fibers")
        print("     TP:", round(stat.mean(CP_res[a][1][0])*100,o), "%")
        print("     FP:", round(stat.mean(CP_res[a][1][1])*100,o), "%")
        print("     FN:", round(stat.mean(CP_res[a][1][2])*100,o), "%")
        
        print("Metrics")
        print("     a:", round(stat.mean(CP_res[a][2][0])*100,o), "%")
        print("     b:", round(stat.mean(CP_res[a][2][1])*100,o), "%")
        print("     c:", round(stat.mean(CP_res[a][2][2])*100,o), "%")
        print("     d:", round(stat.mean(CP_res[a][2][3])*100,o), "%")
        
        a += 1
    
    print("\n")
    
    # Plots
    
    Labels=[
                [r'TP', r'FP', r'FN'],
                [r'TP', r'FP', r'FN'],
                [r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$']
                ]
    Metric = ["Area","Fibers","Performance parameters"]
    for m in M:
        CP_stat = []
        a=0
        while a < len(CP_Algorithms):
            CP_stat.append([[],[],[]]) #plot data form
            n = 0
            while n < len(Labels[m]):
                CP_stat[a][0].append(stat.stdev(CP_res[a][2][n]))
                CP_stat[a][1].append(stat.mean(CP_res[a][2][n]))
                CP_stat[a][2].append(stat.stdev(CP_res[a][2][n]))
                n+=1
            a+=1
        Title = str("Effectiveness based on "+Metric[m])
        PLOT(CP_stat, CP_Algorithms,Title ,Labels[m])

# end
print("\n")
print("\n----- END PROGRAM ----- \n")
T11 = time.time()
print("> " + str(round((T11 - T00), 1)) + "[s] <")
if "draw" in Detail[0]:
    cv.waitKey(1)
