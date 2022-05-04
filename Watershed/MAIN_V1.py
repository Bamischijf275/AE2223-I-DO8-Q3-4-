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


# macro parameters
Loop = "" #Range, Random, List
Name = "Tape_B"

Detail = [["draw","print","save"], 250] #draw/print/save, substep Dt
TypeOUT = [".jpg",".csv"]
Save = ["Img","Matrix","Extra"]

# function parameters
    # WT
WT_Parameters = [3, [0.5, 3, 1], [2, 2, 1], 3, "ellipse"]  # Radius, Relative errors, Filter, kernel

WT_PathIN = "../Data/Tape_B/Images/"
WT_PathOUT = "../Data Processed/Watershed/"
WT_Type = [".jpg",".png",".csv"] #in, out_img, out_matrix

    # CP
CP_Parameters = [0.8] #cutoff, DRAW, Dt

CP_PathIN = "../Data Processed/"
CP_PathOUT = "../Data Processed/Comparator/"
CP_Algorithms = ["Watershed"]
CP_GroundTruth = "Annotated"
CP_Type = [".csv",".png"]


# file get
NAMES = []
if Loop == "Range":
    N = [1,10]
    M = [1,20]

    n = N[0]
    while n <= N[1]:
        m = M[0]
        while m <= M[1]:
            name = Name +"_"+ str(n)  +"_"+ str(m)
            NAMES.append(name)
            m +=1
        n += 1
elif Loop == "List":
    N = [1,1,2,3,5,6,7,8,8,11]
    M = [4,7,3,8,7,6,3,5,9,6]
    i = 0
    while i < len(N):
        name = Name +"_"+ str(N[i])  +"_"+ str(M[i])
        NAMES.append(name)
        i+=1
else:
    m,n = 6,6,
    name = Name +"_"+ str(n)  +"_"+ str(m)
    NAMES.append(name)

# process
path_script = os.path.dirname(__file__)

CP_Confusion = []
a = 0
while a < len(CP_Algorithms):
    CP_Confusion.append([0,0,0,0,0])
    a += 1

for name in NAMES:
    print("\n\n NEWFILE : ", name, "\n")
    T0 = time.time()
    # Waterhsed
    print("\n WATERSHED : \n")
    FileName = name + WT_Type[0]
    path = os.path.join(path_script, WT_PathIN, FileName)
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
        path=str(path+FileName)
        cv.imwrite(path, WT_Image)
        
    if "Matrix" in Save:
        FileName = name + WT_Type[2]
        print("saving WT Matrix : ", FileName)
        path = os.path.join(path_script, WT_PathOUT)
        os.chdir(path)
        path=str(path+FileName)
        pd.DataFrame(WT_Matrix).to_csv((path), header="none", index="none")
        
    if "Extra" in Save:
        IMG = WT[2]
        i = 0
        for Img in IMG:
            img,step = Img
            FileName = name+ "_step_" + str(i) + "_"+ step+ WT_Type[1]
            print("saving extra WT Image : ", FileName)
            path = os.path.join(path_script, WT_PathOUT)
            os.chdir(path)
            path=str(path+"Extras/"+FileName)
            cv.imwrite(path, img)
            i+=1
    
    # Comparator
    print("\n COMPARATOR : \n")
    path = os.path.join(path_script, CP_PathIN, CP_GroundTruth, name + CP_Type[0])
    CP_MatrixT = np.genfromtxt(path, delimiter=",")
    
    a = 0
    for Alg in CP_Algorithms:
        # File
        print(Alg, "against", CP_GroundTruth, ": ")
        
        path = os.path.join(path_script, CP_PathIN, Alg, name + CP_Type[0])
        CP_MatrixR = np.genfromtxt(path, delimiter=",")
        print(path)
        if (Alg == "Watershed") and False:
            CP_MatrixR = WT_Matrix
    
        # Comp
        CP = COMPARATOR(CP_MatrixT, CP_MatrixR, CP_Parameters, Detail)   
        result = CP[0]
        
        # Results
        print("True Fibers:",result[2])
        print("Accuracy TP = CI:", round(100*result[1][0]/result[2],1), "%")
        print("Accuracy TN = / :", round(100*result[1][1]/result[2],1), "%")
        print("Accuracy FP = MI:", round(100*result[1][2]/result[2],1), "%")
        print("Accuracy FN = ND:", round(100*result[1][3]/result[2],1), "%")
        
        CP_Confusion[a][0] += result[1][0]
        CP_Confusion[a][1] += result[1][1]
        CP_Confusion[a][2] += result[1][2]
        CP_Confusion[a][3] += result[1][3]
        CP_Confusion[a][4] += result[2]
        
        # Save images
        if "Extra" in Save:
            IMG = CP[1]
            i = 0
            for Img in IMG:
                img,step = Img
                FileName = name+ "_step_" + str(i) + "_"+ step+ CP_Type[1]
                print("saving extra CP Image : ", FileName)
                path = os.path.join(path_script, CP_PathOUT)
                os.chdir(path)
                path=str(path+FileName)
                cv.imwrite(path, img)
                i+=1
        a += 1
    
    # end of name index[i][j]
    T1 = time.time()
    print("> " + str(round((T1 - T0),1)) + "[s] <")
    
# Algo Stats
print("\n CONCLUSION : \n")
a = 0
while a < len(CP_Algorithms):
    print(CP_Algorithms[a], "against", CP_GroundTruth, ": ")
    print("Accuracy TP = CI:", round(100*CP_Confusion[a][0]/CP_Confusion[a][4],1), "%")
    print("Accuracy TN = / :", round(100*CP_Confusion[a][1]/CP_Confusion[a][4],1), "%")
    print("Accuracy FP = MI:", round(100*CP_Confusion[a][2]/CP_Confusion[a][4],1), "%")
    print("Accuracy FN = ND:", round(100*CP_Confusion[a][3]/CP_Confusion[a][4],1), "%")
    a += 1
    
# end
T11 = time.time()
print("Total Time :")
print("> " + str(round((T11 - T00),1)) + "[s] <")
if "draw" in Detail[0]:
    cv.waitKey(1)
print("\n----- END PROGRAM ----- \n")