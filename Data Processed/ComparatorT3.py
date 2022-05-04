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
    Col_CM = [(0, 255, 0),(150,150,150),(0,0,255),(255,0,0,0)] #TP-G,TN-,FP-B,FN-R
    Col_Background = (100, 100, 100)
    
    Cutoff = PARAMETERS[0]
    Show = PARAMETERS[1]
    ShowTime = PARAMETERS[2]
    
    Result = [0,0,0,0, 0,0,0,0 ,0,0] #TP,TN,FP,FN (pixels, fibers), TrueFib, ResultFib
    Result = [[0,0,0,0],[0,0,0,0],0,0]
    
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
    
    Result[2] += len(FibersT)
    Result[3] += len(FibersR)
    
    #Loop through every TRUE fiber
    Ti = time.time()
    Progress = len(FibersT)
    progress = 0
    print("Fiber Comparison Progress:")
    
    for ID_T in FibersT:
        
        ID_p = -1
        
        progress += 1
        PROGRESS(progress, Progress, prefix='', suffix='', length=30)
        
        #rectangle fiber (truth)
        RectT = SubRect(MatrixT, ID_T)
        
        #find correspodance T to R (ID)
        SubMatrixR = MatrixR[RectT[0]:RectT[2],RectT[1]:RectT[3]]
        Nmax = 0
        for ID in FibersR:
            if MatrixCount(SubMatrixR, ID) >= Nmax:
                ID_R = ID
                Nmax = MatrixCount(SubMatrixR, ID_R)
    
        #rectangle fiber (corr. result)
        if Nmax != 0:
            RectR = SubRect(MatrixR, ID_R)
        else: #no fiber found
            RectR = RectT
        
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
        
        h,w = SubMatrixT.shape
        Area = h*w
        TP = MatrixCount(MUL_Matrix, 1)
        FN = MatrixCount(DIF_Matrix, 1)
        FP = MatrixCount(DIF_Matrix, -1)
        TN = Area - TP-FP-FN
        
        #Results
        Result[0][0]+=TP
        Result[0][1]+=TN
        Result[0][2]+=FP
        Result[0][3]+=FN
        
        if (TP+FP)==0: #none found - FN
            Result[1][3]+=1
        elif TP/(FP+TP) >= Cutoff: #TP
            Result[1][0]+=1
            
        elif FP == 0 or FN == 0: #no overlap - no change
            Result = Result
        elif FP >= FN: #FP
            Result[1][2]+=1
        elif FN > FP:   #FN
            Result[1][3]+=1
            
        else:
            Result[1][1]+=1
            ID_p = ID_T
        
        #Image
        if Show == "DRAW":
            for i in range(len(SubMatrixR)):
                for j in range(len(SubMatrixR[0])):
                    m = MUL_Matrix[i][j]
                    n = DIF_Matrix[i][j]
                    if m == 1: Col=Col_CM[0]
                    elif n == -1: Col=Col_CM[2]
                    elif n == 1 : Col=Col_CM[3]
                    else: Col=Col_CM[1]
                    if ID_T==ID_p: 
                        Col=(255,255,255)
                    cv.circle(img_out, (int(RectTR[1]+j),int(RectTR[0]+i)),0,Col,-1)
            Tf = time.time()
            if Tf-Ti >= ShowTime:
                cv.imshow("Accuracy",img_out)
                Ti = Tf
        #debug
        if ID_T == ID_p and True:
            print("\n DEBUG")
            print("Fiber ID : ", ID_T)
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
    print("> " + str(round((T6 - T0)*1000)) + "[ms] <")
    return(Result)

#FUNCTIONS
def MatrixID(matrix):
    Fibers = np.unique(matrix)
    Fibers.sort()
    Fibers = Fibers[1:]
    
    return(Fibers)
    
def SubRect(matrix, ID):
    place = np.where(matrix==ID)
    rect = [min(place[0]),min(place[1]),max(place[0]),max(place[1])]
    return(rect)
    
def MatrixCount(matrix, ID):
    N = np.count_nonzero(matrix == ID)
    return(N)
    
def MatrixBin(matrix, ID):       
    matrixBIN = np.where(matrix != ID, 0, matrix)
    matrixBIN = np.where(matrix == ID, 1, matrix)
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
print("----- *-* -----")
print("----- START PROGRAM -----")
print("----- *-* -----")

T00 = time.time()
Parameters = [0.8, "", 1] #cutoff, DRAW, Dt
RESULTS = []

#Algos
Algorithms = ["Test","Watershed"]
GroundTruth = "Test"

#files
Name = "Tape_B"
Type=[".csv",".csv"]
N=[2,11]
M=[2,11]
n = N[0]
mm = M[0]

for Alg in Algorithms:
    print("\n ----- * -----")
    print("\n\n NEW ALGORITHM \n")
    print("Comparing : ", Alg, " against ", GroundTruth)
    Dir = [GroundTruth,Alg]
    Results=[0,0,0,0,0]
    
    n = N[0]
    while n <= N[1]:
        m=M[0]
        while m <= M[1]:
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
    
            result = COMPARATOR(MatrixT, MatrixR, Parameters)
            
            #TP,TN,FP,FN
            Results[0] += result[1][0]
            Results[1] += result[1][1]
            Results[2] += result[1][2]
            Results[3] += result[1][3]
            totalFib = result[2]
            Results[4] += totalFib
            
            print("totals T,R:",result[2:])
            print("Accuracy TP = CI:", round(100*result[1][0]/totalFib,1), "%")
            print("Accuracy TN = / :", round(100*result[1][1]/totalFib,1), "%")
            print("Accuracy FP = MI:", round(100*result[1][2]/totalFib,1), "%")
            print("Accuracy FN = ND:", round(100*result[1][3]/totalFib,1), "%")
            
            print("\n ENDFILE \n")
            m+=1
        n+=1
        
    print("\n ----- * -----")
    print("\n Algorithm Results:")
    RESULTS.append(Results)
    print("Accuracy TP = CI:", round(100*Results[0]/Results[4],1), "%")
    print("Accuracy TN = / :", round(100*Results[1]/Results[4],1), "%")
    print("Accuracy FP = MI:", round(100*Results[2]/Results[4],1), "%")
    print("Accuracy FN = ND:", round(100*Results[3]/Results[4],1), "%")
    print("\n ----- * -----")

print("\n ----- *** ----- \n")
print("CONCLUSION :")
a = 0
for Alg in Algorithms:
    print("\n")
    print("Comparing : ", Alg, " against ", GroundTruth)
    print("Accuracy TP = CI:", round(100*RESULTS[a][0]/RESULTS[a][4],1), "%")
    print("Accuracy TN = / :", round(100*RESULTS[a][1]/RESULTS[a][4],1), "%")
    print("Accuracy FP = MI:", round(100*RESULTS[a][2]/RESULTS[a][4],1), "%")
    print("Accuracy FN = ND:", round(100*RESULTS[a][3]/RESULTS[a][4],1), "%")
    a += 1
print("\n ----- *** ----- \n")
T11 = time.time()
print("> " + str(round((T11 - T00),1)) + "[s] <")
print("----- *-* -----")
print("\n----- END PROGRAM ----- \n")
print("----- *-* -----")