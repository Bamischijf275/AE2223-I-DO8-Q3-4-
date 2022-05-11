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

Detail = [["print", "draw", "save"], 250]  # draw/print/save, substep Dt
TypeOUT = [".jpg", ".csv"]
Save = ["", "Matrix", ""] #"Img", "Matrix", "Extra"

Compute = ["","","CP"] #WT,CV,CP
CV = [""]

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
        WT_PathOUT = "../Data Processed/Watershed/"
        WT_Type = [".jpg", ".png", ".csv"]  # in, out_img, out_matrix
        
        if Tape == "Large" or Tape == "Cropped":
            WT_PathIN += "UncroppedImages/"
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
            CV_PathIN = "..\Data Processed\AI results\dataset2\masks"
            CV_PathOUT = "..\Data Processed\AI results\dataset2\masks"
            CV_Type=[".tif",".csv"]
        
            PathIN = os.path.join(path_script, CV_PathIN, name + CV_Type[0])
            PathOUT = os.path.join(path_script, CV_PathOUT, name + CV_Type[1])
            CONVERT_TIFtoCSV(PathIN, PathOUT)
            print("Converted",str(name+CV_Type[0]),"to",str(name+CV_Type[1]))
        
        if "CROP" in CV: #crop matrix   
            CV_PathIN = "../Data Processed/Watershed/"
            CV_PathOUT = "../Data Processed/Watershed/"
            CV_Type=[".tif",".csv"]
            
            FileName = name + CV_Type[1]
            path = os.path.join(path_script, CV_PathIN, FileName)
            Arr = np.genfromtxt(path, delimiter=",")
            
            #Arr_crop = CONVERT_CROP(Arr, 2, 5)
            Arr_crop = CONVERT_CROP2(Arr)
            #print(len(Arr_crop))
            
            path = os.path.join(path_script, CV_PathOUT)
            os.chdir(path)
            print("Cropped : ", Arr.shape,"to", Arr_crop[0].shape)
            
            sep = '-'
            n = name.split(sep, 1)[-1]
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
        
    Progress = len(Names)
    progress = 1
    for name in Names:
        print("\n\n -- NEWFILE : ", name, " - ",progress,"/",Progress," -- \n")
        T0 = time.time()
        
        # parameters
        CP_Parameters = [0.8,0.5]  # cutoff, DRAW, Dt

        CP_PathIN = "../Data Processed/"
        CP_PathOUT = "../Data Processed/Comparator/"
        CP_Algorithms = ["Annotated","AI results/manual/mask_csv","AI results/dataset2/mask_csv","AI results/dataset3/mask_csv","AI results/dataset4/mask_csv", "Watershed"]
        CP_GroundTruth = "Annotated/mask_csv"
        CP_Type = [".csv", ".png"]
        
        CP_Confusion = []
        a = 0
        while a < len(CP_Algorithms):
            CP_Confusion.append([])
            a += 1

        # file
        path = os.path.join(path_script, CP_PathIN, CP_GroundTruth, name + CP_Type[0])
        print(path)
        CP_MatrixT = np.genfromtxt(path, delimiter=",")
    
        # compare each algo for each file
        a = 0
        for Alg in CP_Algorithms:
            # File
            print(Alg, "against", CP_GroundTruth, ": ")

            path = os.path.join(path_script, CP_PathIN, Alg, name + CP_Type[0])
            CP_MatrixR = np.genfromtxt(path, delimiter=",")
            #print(path)
    
            # Comp
            CP = COMPARATOR(CP_MatrixT, CP_MatrixR, CP_Parameters, Detail)
            result = CP[0]
            print("Result : ", result)
            
            CP_Confusion[a].append(result)
    
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
    print("\n RESULTS : \n")
    #CP_Confusion = np.array(CP_Confusion)
    
    #print(CP_Confusion)
    #print(CP_Confusion.shape)
    
    #print(result)
    #print(np.array(result).shape)
    
    #Tfib = CP_Confusion[[0][0][2][0]].sum()
    #Tarea = CP_Confusion[[0][0][2][1]].sum()
    
    #print("total number of True fibers : ", Tfib)
    #print("total True fiber area: ", Tarea)
    
    STAT = []
    a = 0
    Score = []
    while a < len(CP_Algorithms):
            #(min,med,max)*( (ci,mi,nd), (a,b,c,d) )
        print("Confusion matrices : ", CP_Confusion[a])
        stats = [[[],[],[],[]],[[],[],[],[]]] # all results
        for result in CP_Confusion[a]:
            stats[0][0].append(result[0][0])
            stats[0][1].append(result[0][1])
            stats[0][2].append(result[0][2])
            stats[0][3].append(result[0][3])
            
            stats[1][0].append(result[1][0])
            stats[1][1].append(result[1][1])
            stats[1][2].append(result[1][2])
            stats[1][3].append(result[1][3])
        
        print(stats)
        Stat = [ # min,med,max
                [
                    [min(stats[0][0]),stats.mean(stat[0][0]),max(stats[0][0])],
                    [min(stats[0][1]),stats.mean(stat[0][1]),max(stats[0][1])],
                    [min(stats[0][2]),stats.mean(stat[0][2]),max(stats[0][2])],
                    [min(stats[0][3]),stats.mean(stat[0][3]),max(stats[0][3])]
                ],
                [
                    [min(stats[0][0]),stats.mean(stat[0][0]),max(stats[0][0])],
                    [min(stats[0][1]),stats.mean(stat[0][1]),max(stats[0][1])],
                    [min(stats[0][2]),stats.mean(stat[0][2]),max(stats[0][2])],
                    [min(stats[0][3]),stats.mean(stat[0][3]),max(stats[0][3])]
                ]] 
        
        
        print("\n Comparing ",CP_Algorithms[a], "against", CP_GroundTruth, ": ")
        print("Area")
        print("     Accuracy TP=CI:", round(100 * Stat[0][0][1], 2), "%")
        print("     Accuracy FP=MI:", round(100 * Stat[0][1][1], 2), "%")
        print("     Accuracy FP=MUI:", round(100 * Stat[0][2][1], 2), "%")
        print("     Accuracy FN=ND:", round(100 * Stat[0][3][1], 2), "%")
        
        print("Fibers")
        print("     Accuracy a=CI:", round(100 * Stat[1][0][1], 2), "%")
        print("     Accuracy b=MI:", round(100 * Stat[1][1][1], 2), "%")
        print("     Accuracy c=MUI:", round(100 * Stat[1][2][1], 2), "%")
        print("     Accuracy d=ND:", round(100 * Stat[1][3][1], 2), "%")
        
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