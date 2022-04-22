import numpy as np
import pandas as pd
#import imutils
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os
import pandas as pd
import statistics as stat
# from tqdm import tqdm
import sys
import warnings
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

warnings.filterwarnings('ignore')
import time

def COMPARATOR(MatrixT, MatrixR, PARAMETERS):
    #setup
    Col_CM = [(0, 255, 0),(110,110,110),(0,0,255),(255,0,0,0)] #TP,TN,FP,FN
    Col_Background = (100, 100, 100)
    
    Result = [0,0,0,0, 0,0,0,0 ,0,0] #TP,TN,FP,FN (fractions, ID), TrueFib, ResultFib
    
    height, width = MatrixT.shape
    img_out = np.zeros((height, width, 3), np.uint8)
    img_out[:] = Col_Background
    
    #format matrices
    #MatrixT = np.pad(MatrixT, ((0, 2), (0, 2)))
    #MatrixR = np.pad(MatrixR, ((0, 2), (0, 2)))
    MatrixT = MatrixT.astype(int)
    MatrixR = MatrixR.astype(int)
    
    #identify fibers list
    FibersT = MatrixID(MatrixT)
    FibersR = MatrixID(MatrixR)
    
    Result[8] += len(FibersT)
    Result[9] += len(FibersR)
    
    #Loop through every TRUE fiber
    for ID_T in FibersT:
        #rectangle fiber (truth)
        RectT = SubRect(MatrixT, ID_T)
        #find correspodance T to R (ID)
        SubMatrixR = MatrixR[RectT[0]:RectT[2],RectT[1]:RectT[3]]
        
        Nmax = 0
        for ID in FibersR:
            if MatrixCount(SubMatrixR, ID) > Nmax:
                ID_R = ID
                Nmax = MatrixCount(SubMatrixR, ID_R)
    
        #rectangle fiber (corr. result)
        if Nmax != 0:
            RectR = SubRect(MatrixR, ID_R)
        else: #no fiber found
            RectR = RectT
            ID_R = -1
        
        #rectangle fiber (combined)
        RectTR = [0,0,0,0]
        RectTR[0] = min(RectT[0], RectR[0])
        RectTR[1] = min(RectT[1], RectR[1])
        RectTR[2] = max(RectT[2], RectR[2])
        RectTR[3] = max(RectT[3], RectR[3])
        
        #reformat to trimmed binary
        SubMatrixR = MatrixBin((MatrixR[RectTR[0]:RectTR[2],RectTR[1]:RectTR[3]]),ID_R)
        SubMatrixT = MatrixBin((MatrixT[RectTR[0]:RectTR[2],RectTR[1]:RectTR[3]]),ID_T)
        
        #Compare
        DIF_Matrix = np.subtract(SubMatrixT, SubMatrixR)
        MUL_Matrix = np.multiply(SubMatrixT, SubMatrixR)
        
        Area = (RectTR[2]-RectTR[0])*(RectTR[3]-RectTR[1])
        TP = MatrixCount(MUL_Matrix, 1)
        FN = MatrixCount(DIF_Matrix, 1)
        FP = MatrixCount(DIF_Matrix, -1)
        TN = Area - TP-FP-FN
        
        #Results
        Result[0]+=TP
        Result[1]+=TN
        Result[2]+=FP
        Result[3]+=FN
        
        if TP == 0:
                        Result[7]+=1
        elif TP/(FP+FN+TP) >= 0.5:
                        Result[4]+=1
        elif FP > FN:   Result[6]+=1
        elif FN > FP:   Result[7]+=1 
        else:           Result[5]+=1
        
        #Image
        
        
        
        #debug
        if ID_T == 1 and True:
            np.set_printoptions(threshold=sys.maxsize)
            print(ID_T)
            print("Sub-Matrices:")
            print(SubMatrixT)
            print(SubMatrixR)
            print("Ops-Matrices:")
            print(DIF_Matrix)
            print(MUL_Matrix)
            print("result:")
            print(TP,TN,FP,FN)
        
    
    #format results
    return(Result)

#FUNCTIONS
def MatrixID(matrix):
    Fibers = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                Fibers.append(matrix[i][j])
    Fibers = list(dict.fromkeys(Fibers))
    Fibers.sort()
    return(Fibers)
    
def SubRect(matrix, ID):
    #imin, jmin, imax, jmax
    rect = [len(matrix),len(matrix[0]),0,0]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == ID:
                if i < rect[0]:#imin ->
                    rect[0] = i
                elif i > rect[2]:#imax <-
                    rect[2] = i
                if j < rect[1]:#idem j
                    rect[1] = j
                elif j > rect[3]:
                    rect[3] = j
    return(rect)
    
def MatrixCount(matrix, ID):
    N = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == ID:
                N+=1
    return(N)
    
def MatrixBin(matrix, ID):
    matrixBIN = matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != ID: matrixBIN[i][j] = 0
            else: matrixBIN[i][j] = 1
    return(matrixBIN)
    
# MAIN LOOP
print("----- START PROGRAM ----- \n")
T00 = time.time()
#name convention
Dir = ["GroundTruth","Watershed"]
Name = "Tape_B"
Type=[".jpg.tif.csv",".csv"]
N=[1,2,3,4,5,6,7,8,9]
M=[1,2,3,4,5,6,7,8,9,10]
n = 8
m = 9
while n < len(N):
    while m < len(M):
        name = Name+"_"+str(N[n])+"_"+str(M[m])
        print(str(name))
        #extract matrices
        pathScript=os.path.dirname(__file__)
        
        pathT = Dir[0] + "/" + name + Type[0]
        pathT = os.path.join(pathScript,pathT)
        #print(pathT)
        
        pathR = Dir[1] + "/" + name + Type[1]
        pathR = os.path.join(pathScript,pathR)
        #print(pathR)
        
        MatrixT = np.genfromtxt(pathT, delimiter=",")
        MatrixR = np.genfromtxt(pathR, delimiter=",")
        result = COMPARATOR(MatrixT, MatrixR,0.8)
        print("area: ", result[0:4])
        print("fibers :",result[4:8])
        print("totals T,R:",result[8:])
        m+=1
    m=0
    n+=1
T11 = time.time()
print("----- END PROGRAM ----- \n")
print("> " + str(round((T11 - T00),1)) + "[s] <")