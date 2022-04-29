import numpy as np
import pandas as pd
#import imutils
import math
import cv2 as cv
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
np.set_printoptions(threshold=sys.maxsize)
import time

def COMPARATOR(MatrixT, MatrixR, PARAMETERS):

    #SETUP
    #parameters
    T0 = time.time()
    Col_CM = [(0, 255, 0),(110,110,110),(0,0,255),(255,0,0,0)] #TP-G,TN-,FP-B,FN-R
    Col_Background = (100, 100, 100)
    
    Cutoff = PARAMETERS[0]
    Show = PARAMETERS[1]
    ShowTime = PARAMETERS[2]
    
    Result = [0,0,0,0, 0,0,0,0 ,0,0] #TP,TN,FP,FN (pixels, fibers), TrueFib, ResultFib
    
    height, width = MatrixT.shape
    img_out = np.zeros((height, width, 3), np.uint8)
    img_out[:] = Col_Background
    
    #format matrices
    SizeT = MatrixT.shape
    SizeR = MatrixR.shape
    print("Matrix Size T,R Input:", SizeT,SizeR)
    
    MatrixT = MatrixT.astype(int)
    MatrixT = np.delete(MatrixT, (0), axis=0)
    MatrixT = np.delete(MatrixT, (0), axis=1)
    SizeT = MatrixT.shape
    
    MatrixR = MatrixR.astype(int)
    MatrixR = np.delete(MatrixR, (0), axis=0)
    MatrixR = np.delete(MatrixR, (0), axis=1)
    SizeR = MatrixR.shape

    if SizeT != SizeR:
        Matrix = np.zeros(SizeT)
        Matrix[:MatrixR.shape[0],:MatrixR.shape[1]]
        MatrixR = Matrix
    if SizeT != SizeR:
        Matrix = np.zeros(SizeR)
        Matrix[:MatrixT.shape[0],:MatrixT.shape[1]]
        MatrixT = Matrix
    print("Matrix Size T,R Trimmed:", SizeT,SizeR)
    
    #identify fibers list
    FibersT = MatrixID(MatrixT)
    FibersR = MatrixID(MatrixR)
    
    Result[8] += len(FibersT)
    Result[9] += len(FibersR)
    
    #Loop through every TRUE fiber
    if Show =="DRAW":cv.waitKey(1)
    Ti = time.time()
    Progress = len(FibersT)
    progress = 0
    print("Fiber Comparison Progress:")
    
    for ID_T in FibersT:
        
        progress += 1
        PROGRESS(progress, Progress, prefix='', suffix='', length=30)
        
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
        
        if (TP+FP)==0:
            result[3]+=1
        elif TP/(FP+TP) >= Cutoff:
            Result[4]+=1
        elif FP > FN:
            Result[6]+=1
            print(ID_T)
        elif FN > FP:   
            Result[7]+=1
            print(ID_T)
        else:          
            Result[5]+=1
        
        #Image
        if Show == "DRAW":
            for i in range(len(SubMatrixR)):
                for j in range(len(SubMatrixR[0])):
                    m = MUL_Matrix[i][j]
                    n = DIF_Matrix[i][j]
                    if m == 1: Col=Col_CM[0]
                    elif n == -1: Col=Col_CM[2]
                    elif n == 1 : Col=Col_CM[3]
                    else: Col=False
                    if Col != False:
                        if ID_T==61: 
                            Col=(255,255,255)
                        cv.circle(img_out, (int(RectTR[0]+i),int(RectTR[1]+j)),0,Col,-1)
            Tf = time.time()
            if Tf-Ti >= ShowTime:
                cv.imshow("Accuracy",img_out)
                Ti = Tf
        #debug
        if ID_T == 61 and True:
            print(ID_T)
            print("Sub-Matrices:")
            print(SubMatrixT)
            print(SubMatrixR)
            print("Ops-Matrices:")
            print(DIF_Matrix)
            print(MUL_Matrix)
            print("result:")
            print(TP,TN,FP,FN)
        if Show =="DRAW":cv.waitKey(1)
    if Show =="DRAW":cv.waitKey(1)
    
    #format results
    T6 = time.time()
    print("> " + str(round((T6 - T0))) + "[s] <")
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
    
def PROGRESS(iteration, total, prefix='', suffix='', decimals=0, length=10, fill='█', printEnd="\r"):
        # Print Progress bar
        percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration) / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r  {prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print('\n')
    
# MAIN LOOP
print("----- START PROGRAM ----- \n")
T00 = time.time()
#name convention

if False: #Cropped images
    Dir = ["Test","Watershed"]
    Name = "Tape_B"
    Type=[".csv",".csv"]
    N=10
    M=10
    n = 1
    mm = 1
else: #Full Tapes
    Dir = ["Test","Test"]
    Name = "Tape_B"
    Type=[".csv",".csv"]
    N=2
    M=1
    n = 2
    mm = 1
    
Results=[0,0,0,0]

while n <= N:
    m=mm
    while m <= M:
        print("\n NEWFILE")
        name = Name+"_"+str(n)+"_"+str(m)
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
        #MatrixR = np.loadtxt(open(pathT, "rb"), delimiter=",")

        result = COMPARATOR(MatrixT, MatrixR, [0.8,"DRAW",1])
        
        #TP,TN,FP,FN
        Results[0] += result[4]
        Results[1] += result[5]
        Results[2] += result[6]
        Results[3] += result[7]
        totalFib = result[4]+result[6]
        
        print("area: ", result[0:4])
        print("fibers :",result[4:8])
        print("totals T,R:",result[8:])
        print("Accuracy TP:", round(result[4]/totalFib,3), "%")
        print("Accuracy TN:", round(result[5]/totalFib,3), "%")
        print("Accuracy FP:", round(result[6]/totalFib,3), "%")
        print("Accuracy FN:", round(result[7]/totalFib,3), "%")
        
        print("\n ENDFILE \n")
        m+=1
    n+=1
print("\n\n --- STATS --- \n")
print(Results)

T11 = time.time()
print("----- END PROGRAM ----- \n")
print("> " + str(round((T11 - T00),1)) + "[s] <")