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
import imutils
import math
import numpy as np
import numpy.random as rnd
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import os
import pandas as pd
import statistics as stat
# from tqdm import tqdm
import sys
import warnings
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import PIL

# own functions
from FUNCTIONS_V1 import *

# init
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)

# MAIN
print("\n----- START PROGRAM ----- \n")

# Macro parameters
Loop = "Random"  # Range, Random, List, All
N,M,K = [],[],[50,1]
Name = "Tape_B"
Tape= "Large" #Large, Cropped, none=smalls

Detail = [["", "", "save"], 250]  # draw/print/save, substep Dt
TypeOUT = [".jpg", ".csv"]
Save = ["", "Matrix", ""]

Compute = ["WT","",""] #WT,CP,CV
CV = ["CROP"]

errorMin = 10**(-3)

# file names


# process
path_script = os.path.dirname(__file__)

# Waterhsed
if "WT" in Compute:
    print("\n WATERSHED : \n")
    
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
        
        # parameters
        WT_Parameters = [2.5, [0.5, 5, 2], [5, 20, 2], 3, "exact","UF"]  # Radius, Relative errors, Filter, kernel
        #WT_Parameters = [4.5, [0.4, 5, 2], [6, 48, 2], 3, "exact",""]  # <- Combined Score, Fiber ID ^

        WT_PathIN = "../Data/Tape_B/"
        WT_PathOUT = "../Data Processed/Watershed/Training/"
        WT_Type = [".jpg", ".png", ".csv"]  # in, out_img, out_matrix
        
        if Tape == "Large" or Tape == "Cropped":
            WT_PathIN += "Tape_B_2_JPG/"
        else: 
            WT_PathIN += "Images/"
        
        # file
        FileName = name + WT_Type[0]
        path = os.path.join(path_script, WT_PathIN, FileName)
        #print(path)
        WT_Image = cv.imread(path)
        #print(path)
    
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
            pd.DataFrame(WT_Matrix).to_csv((path), header="none", index="none")
    
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
    Names = list(dict.fromkeys(Names))
    print("Filenames :")
    for name in Names:
        print(name)
        
    Progress = len(Names)
    progress = 1
    for name in Names:
        print("\n\n -- NEWFILE : ", name, " - ",progress,"/",Progress," -- \n")
        T0 = time.time()
        
        # parameters
        CV_PathIN = ""
        CV_PathOUT = "../Data Processed/Watershed/"
        CV_Type=[".tif",".csv"]
        
        # various converters
        if "TIFtoCSV" in CV:
            path = os.path.join(path_script, CV_PathIN, name + CV_Type[0])
            CONVERT_TIFtoCSV(CV_PathIN, CV_PathOUT)
        
        if "CROP" in CV: #crop matrix     
            FileName = name + CV_Type[1]
            print(FileName)
            path = os.path.join(path_script, CV_PathOUT, FileName)
            Arr = np.genfromtxt(path, delimiter=",")
            
            Arr_crop = CONVERT_CROP(Arr, 5, 2)
            path = os.path.join(path_script, CV_PathOUT)
            os.chdir(path)
            
            sep = '-'
            n = name.split(sep, 1)[-1]
            m = 1
            for Arr in Arr_crop:
                FileName = Name + "_" + str(n) + "_" + str(m)
                print("saving :",FileName)
                path = os.path.join(path_script, CV_PathOUT, FileName+CV_Type[1])
                pd.DataFrame(Arr).to_csv((path), header="none", index="none")
                m += 1
                
        progress += 1
        T1 = time.time()
        print("> " + str(round((T1 - T0), 1)) + "[s] <")
                
# Comparator
if "CP" in Compute:
    print("\n COMPARATOR : \n")
    
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
        
        # parameters
        CP_Parameters = [0.8]  # cutoff, DRAW, Dt

        CP_PathIN = "../Data Processed/"
        CP_PathOUT = "../Data Processed/Comparator/"
        CP_Algorithms = ["Watershed"]
        CP_GroundTruth = "Annotated"
        CP_Type = [".csv", ".png"]
        
        CP_Confusion = []
        a = 0
        while a < len(CP_Algorithms):
            CP_Confusion.append([[0,0,0,0,0],[0, 0, 0, 0, 0, 0]]) # Fibers
            a += 1

        # file
        path = os.path.join(path_script, CP_PathIN, CP_GroundTruth, name + CP_Type[0])
        CP_MatrixT = np.genfromtxt(path, delimiter=",")
    
        # compare each algo for each file
        a = 0
        for Alg in CP_Algorithms:
            # File
            print(Alg, "against", CP_GroundTruth, ": ")
            
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
            if result[2] == 0:
                result[2] += errorMin
            print("Result : ", result)
            
            # Area
            CP_Confusion[a][0][0] += result[0][0]
            CP_Confusion[a][0][1] += result[0][1]
            CP_Confusion[a][0][2] += result[0][2]
            CP_Confusion[a][0][3] += result[0][3]
            if result[4] == 0:
                CP_Confusion[a][0][4] += errorMin
            else:
                CP_Confusion[a][0][4] += result[4]
            # Fibers
            CP_Confusion[a][1][0] += result[1][0]
            CP_Confusion[a][1][1] += result[1][1]
            CP_Confusion[a][1][2] += result[1][2]
            CP_Confusion[a][1][3] += result[1][3]
            CP_Confusion[a][1][4] += result[2]
            CP_Confusion[a][1][5] += result[3]
    
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
             
    # end of name index[i][j]
    progress += 1
    T1 = time.time()
    print("> " + str(round((T1 - T0), 1)) + "[s] <")

if "CP" in Compute:
    # Algo Stats
    print("\n CONCLUSION : \n")
    print("total number of True fibers : ", CP_Confusion[0][1][4])
    print("total number of fibers found: ", CP_Confusion[0][1][5])
    print("total True Area: ", CP_Confusion[0][0][4])
    a = 0
    Score = []
    while a < len(CP_Algorithms):
        print("\n Comparing ",CP_Algorithms[a], "against", CP_GroundTruth, ": ")
        print("Fibers")
        print("     Accuracy TP = CI:", round(100 * CP_Confusion[a][1][0] / CP_Confusion[a][1][4], 2), "%")
        #print("     Accuracy TN = / :", round(100 * CP_Confusion[a][1][1] / CP_Confusion[a][1][4], 2), "%")
        print("     Accuracy FP = MI:", round(100 * CP_Confusion[a][1][2] / CP_Confusion[a][1][4], 2), "%")
        print("     Accuracy FN = ND:", round(100 * CP_Confusion[a][1][3] / CP_Confusion[a][1][4], 2), "%")
        
        print("Area")
        print("     Accuracy TP = CI:", round(100 * CP_Confusion[a][0][0] / CP_Confusion[a][0][4], 2), "%")
        #print("     Accuracy TN = / :", round(100 * CP_Confusion[a][0][1] / CP_Confusion[a][0][4], 2), "%")
        print("     Accuracy FP = MI:", round(100 * CP_Confusion[a][0][2] / CP_Confusion[a][0][4], 2), "%")
        print("     Accuracy FN = ND:", round(100 * CP_Confusion[a][0][3] / CP_Confusion[a][0][4], 2), "%")
        
        Score.append( 
                (CP_Confusion[a][0][0] / (CP_Confusion[a][0][4])) * 
                     (CP_Confusion[a][1][0] / CP_Confusion[a][1][4]) 
                     )
        
        a += 1
        
    print("\n")
    a = 0
    while a < len(CP_Algorithms):
        print("Score of ",CP_Algorithms[a], " :     ", round(100*Score[a],3) ,"[%]")
        a += 1

# end
print("\n")
print("\n----- END PROGRAM ----- \n")
T11 = time.time()
print("> " + str(round((T11 - T00), 1)) + "[s] <")
if "draw" in Detail[0]:
    cv.waitKey(1)
