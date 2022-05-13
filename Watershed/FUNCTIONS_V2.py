# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:32:55 2022

@author: huege
"""

import PIL
# STANDARD MODULES
import cv2 as cv
import imutils
import math
import numpy as np
import numpy.random as rnd
import pandas as pd
import statistics as stat
import time
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import os
import matplotlib.pyplot as plt

# FUNCTIONS

# ! Watershed !
def WATERSHED(IMAGE, PARAMETERS, DETAIL):
    if "print" in DETAIL[0]:
        print("\n init :")
        T0 = time.time()

    # PARAMETERS:

    # program
    P_SSp = DETAIL[1]
    extra_IMGS = []

    # parameters
    F_mean = PARAMETERS[0]
    D_RE, F_RE, E_RE = PARAMETERS[1]
    PyrFilt1, PyrFilt2, PyrFiltIT = PARAMETERS[2]  # PMSF - iterations,spatial radius,color radius
    ke = PARAMETERS[3]
    P_shape = PARAMETERS[4]
    P_filter = PARAMETERS[5]

    R_min = F_mean * (1 - D_RE)
    MinDist = math.floor(R_min)

    # SETUP

    # Display
    Col_ShapeCenter = (255, 255, 255)
    Col_Boundary = (0, 255, 0)
    Col_Background = (100, 100, 100)

    # Init Print
    if "print" in DETAIL[0]:
        print('INITIAL PARAMETERS')
        print("[INFO] mean fiber radius     :", format(round(F_mean, 3)))
        print("[INFO] distance T. factor    :", format(round(D_RE * 100)), "%")
        print("[INFO] radius error factor   :", format(round(F_RE, 3)))
        if P_shape == "ellipse":
            print("[INFO] fiber ecc. error      :", format(round(E_RE, 3)))
        print("[INFO] PMS filter            :", PyrFiltIT, PyrFilt1, PyrFilt2)
        print("[INFO] Noise Kernel Radius   :", format(ke))

        T1 = time.time()
        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")

    # IMAGE PROCESS
    # Open image
    if "print" in DETAIL[0]:
        print("pre-processing:")
        T0 = time.time()

    img = IMAGE.copy()
    if "draw" in DETAIL[0]:
        cv.imshow('Input', img)
        extra_IMGS.append([img, 'IN'])
        cv.waitKey(1)
    height, width, _ = img.shape

    # Pre-processing

    # PMSF
    if "print" in DETAIL[0]:
        print("     Pyramid Mean Shift Filter ...")

    p = PyrFiltIT
    imgPMSF = img.copy()
    while p > 0:
        p -= 1
        imgPMSF = cv.pyrMeanShiftFiltering(imgPMSF, PyrFilt1, PyrFilt2)
        if "print" in DETAIL[0]:
            PROGRESS(PyrFiltIT - p, PyrFiltIT, prefix='', suffix='', length=30)

    if "draw" in DETAIL[0]:
        cv.imshow("PMSF", imgPMSF)
        extra_IMGS.append([imgPMSF.copy(), 'PMSF'])
        cv.waitKey(1)

    # Otsu
    if "print" in DETAIL[0]:
        print("     Otsu binarization ...")

    imgGray = cv.cvtColor(imgPMSF, cv.COLOR_BGR2GRAY)
    imgTSH = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    if "draw" in DETAIL[0]:
        cv.imshow("Thresh", imgTSH)
        extra_IMGS.append([imgTSH, 'TSH'])
        cv.waitKey(1)

    # erode-dilate
    if "print" in DETAIL[0]:
        print("     Noise Reduction ...")

    if ke > 0 and isinstance(ke, int):
        kernel = np.ones((ke, ke), np.uint8)
        imgKer = cv.dilate(cv.erode(imgTSH.copy(), kernel), kernel)
    else:
        imgKer = imgTSH.copy()

    if "draw" in DETAIL[0]:
        cv.imshow("Kernel", imgKer)
        extra_IMGS.append([imgKer.copy(), 'Ker'])
        cv.waitKey(1)

    # End of Pre-processing
    if "print" in DETAIL[0]:
        T1 = time.time()
        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")

    # WATERSHED
    if "print" in DETAIL[0]:
        print("WATERSHED : ")
        T0 = time.time()

    # labelling
    if "print" in DETAIL[0]:
        print("     Labelling ...")
    D = ndimage.distance_transform_edt(imgKer)
    localMax = peak_local_max(D, indices=False, min_distance=MinDist, labels=imgKer)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=imgKer)

    R = A = B = set([])  # shapes stats (radius)
    Shapes = []
    Boundaries = []
    arr_out = np.zeros((height, width), dtype=int)
    img_out = np.zeros((height, width, 3), np.uint8)
    img_out[:] = Col_Background
    imgBnd = IMAGE.copy()
    imgShp = IMAGE.copy()

    # identification
    if "print" in DETAIL[0]:
        print("     identification and shape fitting ...")
        Progress = len(np.unique(labels))
        progress = 0
    if "draw" in DETAIL[0]:
        SSp_0 = time.time()

    # Shape Fitting Loop
    for label in np.unique(labels):
        if "print" in DETAIL[0]:
            progress += 1
            PROGRESS(progress, Progress, prefix='', suffix='', length=30)
        # eliminate background
        if label == 0:
            continue

        # mask and label region
        mask = np.zeros(imgKer.shape, dtype="uint8")
        mask[labels == label] = 255

        # find contours
        Cnts = cv.findContours(mask.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(Cnts)
        c = max(cnts, key=cv.contourArea)

        cntsX = cv.findContours(imgKer.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cntsX = imutils.grab_contours(cntsX)
        for (i, cX) in enumerate(cnts):
            Boundaries.append([cX])
            if "draw" in DETAIL[0]:
                imgBnd = cv.drawContours(imgBnd, [cX], -1, Col_Boundary, 1)

        # SHAPE FITTING

        # draw enclosing circle 
        ID = 1
        if P_shape == "circle":
            ((x, y), r) = cv.minEnclosingCircle(c)
            Col = COL(x, y, r, ID, width, height)
            ID += 1
            Shapes.append([x, y, r])
            if "draw" in DETAIL[0]:
                imgShp = cv.circle(imgShp, (int(x), int(y)), int(r), Col, 1)

            Shapes.append([x, y, r])
            R.add(r)

        # draw inscribed-bound ellipse - used for 'exact' discriminant
        if (P_shape == "ellipse") or (P_shape =="exact"):
            # bounding box
            rect = (x, y), (w, h), Agl = cv.minAreaRect(c)
            Col = COL(x, y, Agl, ID, width, height)
            ID += 1
            box = cv.boxPoints(rect)
            box = np.int0(box)
            
            if "draw" in DETAIL[0]:
                cv.drawContours(imgShp, [box], 0, Col, 1)
                if P_shape == "ellipse":
                    cv.ellipse(imgShp, ((x, y), (w, h), Agl), Col, 1)
                    
            # compute eccentricty
            if w == 0:
                c = (h ** 2) ** (1 / 2)
                e = 100
                A.add(w)
                B.add(h)
            elif w >= h:
                c = ((w ** 2) - (h ** 2)) ** (1 / 2)
                e = c / w
                A.add(w)
                B.add(h)
            else:
                c = ((h ** 2) - (w ** 2)) ** (1 / 2)
                e = c / h
                A.add(h)
                B.add(w)
            
            Shapes.append([x, y, w, h, Agl, c, e])

        # draw
        if "draw" in DETAIL[0]:
            cv.circle(imgShp, (int(Shapes[-1][0]),int(Shapes[-1][1])), 1, Col_ShapeCenter, -1)
            
            SSp_1 = time.time()
            D_SSp = (SSp_1 - SSp_0) * 1000
            if D_SSp > P_SSp:
                SSp_0 = SSp_1
                cv.imshow("Boundaries", imgBnd)
                cv.imshow("Shapes", imgShp)
                cv.waitKey(1)

    # End of Watershed + Shape Fitting
    if "draw" in DETAIL[0]:
        cv.imshow("Boundaries", imgBnd)
        cv.imshow("Shapes", imgShp)

        extra_IMGS.append([imgBnd, 'Bnd'])
        extra_IMGS.append([imgShp, 'Shp'])
        cv.waitKey(1)

    if "print" in DETAIL[0]:
        T1 = time.time()
        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")

    # FIBER SELECTION - OUTLIER
    if "print" in DETAIL[0]:
        print("FIBER IDENTIFICATION :")
        T0 = time.time()

        print("     Initial Stats ...")

    # setup
    if P_shape == "circle":
        R_med = stat.median(R)
        R_avg = stat.mean(R)
    elif (P_shape == "ellipse") or (P_shape == "exact"):
        A_med = stat.median(A)
        A_avg = stat.mean(A)
        B_med = stat.median(B)
        B_avg = stat.mean(B)

    F = set([])  # same but filtered
    Fibers = []
    imgFib = IMAGE.copy()
    ID = 1

    # infos
    if "print" in DETAIL[0]:
        print("     Shape sorting ...")

        Progress = len(Shapes)
        progress = 1
    if "draw" in DETAIL[0]:    
        SSp_0 = time.time()

    # Discriminant Loop
    s = 0
    for shape in Shapes:
        if "print" in DETAIL[0]:
            PROGRESS(progress, Progress, prefix='', suffix='', length=30)
            progress += 1

        # circle
        x, y = shape[0], shape[1]
        if P_shape == "circle":
            r = shape[2]
            if R_med / F_RE < r < R_med * F_RE:
                F.add(r)
                Col = COL(x, y, r, ID, width, height)
                ID += 1
                cv.circle(arr_out, (int(x), int(y)), int(r), ID, -1)

                if "draw" in DETAIL[0]:
                    cv.circle(imgFib, (int(x), int(y)), int(r), Col, -1)
                    cv.circle(img_out, (int(x), int(y)), int(r), Col, -1)

                # fiber ID and appending
                Fibers.append([len(F), round(x), round(y), round(r, 3)])

        # ellipse - used for 'exact' discriminant
        if (P_shape == "ellipse" ) or (P_shape == "exact"):
            a, b, Agl, c, e = shape[2:]
            if (abs(e) < E_RE and B_med / F_RE < min(a, b) and max(a, b) < A_med * F_RE) or (P_filter == "UF"):
                F.add(c)
                Col = COL(x, y, c, ID, width, height)
                ID += 1
                if P_shape == "ellipse":
                    cv.ellipse(arr_out, ((x, y), (a, b), Agl), ID, -1)
                else:
                    cv.drawContours(arr_out, Boundaries[s], -1, ID, -1)

                if "draw" in DETAIL[0]:
                    if P_shape == "ellipse":
                        cv.ellipse(imgFib, ((x, y), (a, b), Agl), Col, -1)
                        cv.ellipse(img_out, ((x, y), (a, b), Agl), Col, -1)
                    else:
                        cv.drawContours(imgFib, Boundaries[s], -1, Col, -1)
                        cv.drawContours(img_out, Boundaries[s], -1, Col, -1)

                # fiber ID and appending
                if P_shape == "ellipse":
                    Fibers.append([len(F), round(x), round(y), round(c, 3), round(e, 3)])
                else:
                    Fibers.append([s, Boundaries[s]])
        # draw       
        if "draw" in DETAIL[0]:
            SSp_1 = time.time()
            D_SSp = (SSp_1 - SSp_0) * 1000
            if (D_SSp > P_SSp):
                SSp_0 = SSp_1
                cv.imshow("Fibers", imgFib)
                cv.imshow("OUTPUT", img_out)
                cv.waitKey(1)
                
        s += 1

    if "draw" in DETAIL[0]:
        cv.imshow("Fibers", imgFib)
        cv.imshow("OUTPUT", img_out)

        extra_IMGS.append([imgFib, 'Fib'])
        extra_IMGS.append([img_out, 'Out'])
        cv.waitKey(1)

    if "print" in DETAIL[0]:
        T1 = time.time()
        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")

    # OUTPUT
    # info
    if "print" in DETAIL[0]:
        print("OUTPUT")
        print("     statisctics ...")
        T0 = time.time()

        if P_shape == ("ellipse"or"exact"):
            E = []
            for ellipse in Fibers:
                E.append(ellipse[-1])
            E_avg = stat.mean(E)
            E_med = stat.median(E)

        # print stats

        print('WATERSHED')
        print("[INFO] unique contours found :", format(len(labels)))
        print("[INFO] unique shapes found   :", format(max(len(R), len(A))))
        print("[INFO] unique fibers found   :", format(len(F)))
        print("-----")

        print('MEAN VALUES')
        if P_shape == "circle":
            print("[INFO] radius    :", format(R_avg))
        elif P_shape == "ellipse":
            print("[INFO] semi-major axis   :", format(round(A_avg / 2, 3)))
            print("[INFO] semi-minor axis   :", format(round(B_avg / 2, 3)))
            print("[INFO] eccentricity      :", format(round(E_avg, 3)))
        print("-----")

        print('MEDIAN VALUES')
        if P_shape == "circle":
            print("[INFO] radius   :", format(R_med))
        elif P_shape == "ellipse":
            print("[INFO] semi-major axis   :", format(round(A_med / 2, 3)))
            print("[INFO] semi-minor axis   :", format(round(B_med / 2, 3)))
            print("[INFO] eccentricity      :", format(round(E_med, 3)))

        T1 = time.time()
        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")
        print("\n")
    return img_out, arr_out, extra_IMGS


# Comparator !
def COMPARATOR(MatrixT, MatrixR, PARAMETERS, DETAIL):
    # SETUP
    CheckForMUI = True
    # parameters
    if "print" in DETAIL[0]:
        print("\n init :")
        T0 = time.time()

    Col_CM = [(0, 255, 0), (150, 150, 150), (0, 0, 255), (255, 0, 0, 0)]  # TP-G,TN-,FP-B,FN-R
    Col_Background = (100, 100, 100)

    Cutoff = PARAMETERS[0]
    ShowTime = DETAIL[1] / 1000
    extra_IMGS = []

    Result = [
            [0, 0, 0, 0], # area: TP,FP,FN , Tarea
            [0, 0, 0, 0], # fibers: TP,FP,FN , Tfib
            [0, 0, 0, 0]  # metrics: A,B,C,D
            ]
    MUI = 0
    CutoffMUI = PARAMETERS[1]

    # format matrices
    MatrixT = np.array(MatrixT)
    MatrixR = np.array(MatrixR)
    
    SizeT = MatrixT.shape
    SizeR = MatrixR.shape
    if "print" in DETAIL[0]:
        print("     Matrix Size T,R Input:", SizeT, SizeR)

    #MatrixT = MatrixT.astype(int)
    MatrixT = np.delete(MatrixT, (0), axis=0)
    MatrixT = np.delete(MatrixT, (0), axis=1)

    SizeT = MatrixT.shape

    #MatrixR = MatrixR.astype(int)
    MatrixR = np.delete(MatrixR, (0), axis=0)
    MatrixR = np.delete(MatrixR, (0), axis=1)
    SizeR = MatrixR.shape
    
    if "print" in DETAIL[0]:
        print("     Matrix Size T,R Input:", SizeT, SizeR)
    
    while MatrixT.shape != MatrixR.shape:
        SizeR = MatrixR.shape
        SizeT = MatrixT.shape
        if SizeR[0] < SizeT[0]:
            #trim T0
            MatrixT = np.delete(MatrixT, (0), axis=0)
        elif SizeR[0] > SizeT[0]:
            #trim R0
            MatrixR = np.delete(MatrixR, (0), axis=0)
        if SizeR[1] < SizeT[1]:
            #trim T1
            MatrixT = np.delete(MatrixT, (0), axis=1)
        elif SizeR[1] > SizeT[1]:
            #trim R1
            MatrixR = np.delete(MatrixR, (0), axis=1)
        #print(SizeR,SizeT)
            
    if "print" in DETAIL[0]:
        print("     Matrix Size T,R Trimmed:", SizeT, SizeR)

    # draw
    if "draw" in DETAIL[0]:
        imgT = np.where(MatrixT != 0, 255, MatrixT)
        extra_IMGS.append([imgT, 'Truth'])
        cv.imshow("Truth", imgT)
        
        imgR = np.where(MatrixR != 0, 255, MatrixR)
        extra_IMGS.append([imgR, 'Algo'])
        cv.imshow("Algo", imgR)

        height, width = MatrixT.shape
        imgConf = np.zeros((height, width, 3), np.uint8)
        imgConf[:] = Col_Background
        cv.waitKey(1)

    # identify fibers list
    FibersT = MatrixID(MatrixT).tolist()
    FibersR = MatrixID(MatrixR).tolist()
    
    Result[1][3] = len(FibersT)
    
    if "print" in DETAIL[0]:
        print("     Fibers in T,R: ", Result[2:])

    # Loop through every TRUE fiber
    if "print" in DETAIL[0]:
        T1 = time.time()
        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")

        T0 = T1
        print("Fiber Comparison Progress :")

    Ti = time.time()
    Progress = len(FibersT)
    progress = 0
    for ID_T in FibersT:

        ID_p = -1
        if "print" in DETAIL[0]:
            progress += 1
            PROGRESS(progress, Progress, prefix='', suffix='', length=30)

        # rectangle fiber (truth)
        RectT = SubRect(MatrixT, ID_T)

        # find correspodance T to R (ID)
        SubMatrixR = MatrixR[RectT[0]:RectT[2], RectT[1]:RectT[3]]
        
        Nmax = 0
        for ID in FibersR:
            n_id = MatrixCount(SubMatrixR, ID)
            if n_id >= Nmax:
                ID_R = ID
                Nmax = MatrixCount(SubMatrixR, ID_R)

        # rectangle fiber (corr. result)
        if Nmax != 0:
            RectR = SubRect(MatrixR, ID_R)
        else:  # no fiber found
            RectR = RectT

        # rectangle fiber (combined)
        RectTR = [0, 0, 0, 0]
        RectTR[0] = min(RectT[0], RectR[0])
        RectTR[1] = min(RectT[1], RectR[1])
        RectTR[2] = max(RectT[2], RectR[2])
        RectTR[3] = max(RectT[3], RectR[3])

        # reformat to trimmed binary
        SubMatrixR = MatrixBin((MatrixR[RectTR[0]:RectTR[2], RectTR[1]:RectTR[3]]), ID_R)
        SubMatrixT = MatrixBin((MatrixT[RectTR[0]:RectTR[2], RectTR[1]:RectTR[3]]), ID_T)

        # Compare
        DIF_Matrix = np.subtract(SubMatrixT, SubMatrixR)
        MUL_Matrix = np.multiply(SubMatrixT, SubMatrixR)

        Tarea = MatrixCount(SubMatrixT, 1)
        TP = MatrixCount(MUL_Matrix, 1)
        FN = MatrixCount(DIF_Matrix, 1)
        FP = MatrixCount(DIF_Matrix, -1)

        # Results
        Result[0][0] += TP
        Result[0][1] += FP
        Result[0][2] += FN
        Result[0][3] += Tarea

        if (TP + FP) == 0:  # none found - FN
            Result[1][2] += 1
            
        elif TP / (FP + TP) >= Cutoff:  # TP
            Result[1][0] += 1
            FibersR.remove(ID_R)

        elif FP != 0 or FN != 0:
            
            if CheckForMUI: # check MUI TO DO
                # find every fiber in SubMatrix
                ID_Rarr = []
                for ID in FibersR:
                    n_id = MatrixCount(SubMatrixR, ID)
                    if n_id != 0:
                        ID_Rarr.append(ID)
                # if the combination of every fiber is CI, then MUI

                if TP / Tarea >= CutoffMUI:  # CI+ == MUI
                    MUI += 1
                
                    for ID in ID_Rarr:
                        Result[1][1] += 1
                        FibersR.remove(ID)
        
            elif FP >= FN:  # FP
                Result[1][1] += 1
                FibersR.remove(ID_R)
            elif FN > FP:  # FN
                Result[1][2] += 1
                
        else: # nothing detected
            Result = Result

        # Image
        if "draw" in DETAIL[0]:
            for i in range(len(DIF_Matrix)):
                for j in range(len(DIF_Matrix[0])):
                    m = MUL_Matrix[i][j]
                    n = DIF_Matrix[i][j]
                    if m == 1:
                        Col = Col_CM[0]
                    elif n == -1:
                        Col = Col_CM[2]
                    elif n == 1:
                        Col = Col_CM[3]
                    else:
                        Col = Col_CM[1]
                    if ID_T == ID_p:
                        Col = (255, 255, 255)
                    cv.circle(imgConf, (int(RectTR[1] + j), int(RectTR[0] + i)), 0, Col, -1)
            Tf = time.time()
            if Tf - Ti >= ShowTime:
                cv.imshow("Confusion", imgConf)
                Ti = Tf
        # debug
        if (ID_T == ID_p) and ("print" in DETAIL[0]):
            print("\n DEBUG")
            print("Fiber ID : ", ID_T)
            print("Sub-Matrices:")
            print(SubMatrixT)
            print(SubMatrixR)
            print("Ops-Matrices:")
            print(DIF_Matrix)
            print(MUL_Matrix)
            print("result:")
            print(TP, FP, FN)

    if "draw" in DETAIL[0]:
        cv.imshow("Accuracy", imgConf)
        extra_IMGS.append([imgConf, 'Confusion'])
        cv.waitKey(1)
    
    # FP
    FP_fib = 0
    for ID_R in FibersR:
        FP_fib += MatrixCount(MatrixR, ID_R)
    Result[0][1] = FP_fib
    Result[1][1] = len(FibersR)
    
    # derive metrics
    Result[2][0] = Result[1][0]/Result[1][3]
    Result[2][1] = Result[1][1]/(Result[1][0]+Result[1][1])
    Result[2][2] = Result[1][2]/Result[1][3]
    Result[2][3] = MUI/Result[1][3]

    if "print" in DETAIL[0]:
        T1 = time.time()
        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")
    return (Result, extra_IMGS)


# others

def COL(x, y, z, ID, width, height):
    # X = int((x / width) * 255)
    # Y = int((y / height) * 255)
    # Z = int(z ** 2)
    rnd.seed(ID)
    X = rnd.randint(0, 255)
    Y = rnd.randint(0, 255)
    Z = rnd.randint(0, 255)
    Col = (X, Y, Z)
    return Col

    # CP


def MatrixID(matrix):
    Fibers = np.unique(matrix)
    Fibers.sort()
    Fibers = Fibers[1:]

    return (Fibers)


def SubRect(matrix, ID):
    place = np.where(matrix == ID)
    rect = [min(place[0]), min(place[1]), max(place[0]), max(place[1])]
    return (rect)


def MatrixCount(matrix, ID):
    N = np.count_nonzero(matrix == ID)
    return (N)


def MatrixBin(matrix, ID):
    matrixBIN = np.where(matrix != ID, 0, matrix)
    matrixBIN = np.where(matrix == ID, 1, matrix)
    return (matrixBIN)

    # QoL


def PROGRESS(iteration, total, prefix='', suffix='', decimals=0, length=10, fill='█', printEnd="\r"):
    # Print Progress bar
    percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration) / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r  {prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print('')
        
def NAMES(loop, N=[], M=[],K=[], tape="", Name="Tape_B"):
    Names = []
    if loop == "Range":
        if N == [] or M==[]: # default
            N = [1, 20]
            M = [1, 10]
            
        n = N[0]
        while n <= N[1]:
            m = M[0]
            while m <= M[1]:
                if tape == "Large" or tape == "Cropped":
                    name = Name + "_2_-" + str(n)
                else:
                    name = Name + "_" + str(n) + "_" + str(m)
                Names.append(name)
                m += 1
            n += 1
            
    elif loop == "List":
        if N == [] or M==[]: # default
            N = [1, 1, 2, 3, 5, 6, 7, 8, 8, 11]
            M = [4, 7, 3, 8, 7, 6, 3, 5, 9, 6]
        i = 0
        while i < len(N):
            if tape == "Large" or tape == "Cropped":
                name = Name + "_2_" + str(N[i])
            else:
                name = Name + "_" + str(N[i]) + "_" + str(M[i])
            Names.append(name)
            i += 1
            
    elif loop == "Random":
        if N == [] or M==[]: # default
            if tape == "Large" or tape == "Cropped":
                N= [0,2197]
            else:
                N,M = [0,20],[0,10]
        elif K == 0:
            K= [20,1]
                
        rnd.seed(K[1])
        
        i = 0
        while i <= K[0]:
            if tape == "Large" or tape == "Cropped":
                name = Name + "_2_" + str(rnd.randint(N[0], N[1]))
            else:
                name = Name + "_" + str(rnd.randint(N[0], N[1])) + "_" + str(rnd.randint(M[0], M[1]))
            Names.append(name)
            i += 1
        
    else:
        if N == [] and M==[]: # default
            n, m = 1,1
        else:
            n,m = N[0],M[0]
            
        if tape == "Large" or tape == "Cropped":
            name = Name + "_2_" + str(n)
        else:
            name = Name + "_" + str(n) + "_" + str(m)
        Names.append(name)
    
    return Names


def CONVERT_TIFtoCSV(pathIN, pathOUT):
    Img = PIL.Image.open(pathIN)
    Arr = np.array(Img)
    if pathOUT != "":
        #pd.DataFrame(Arr).to_csv((pathOUT), header="none", index="none")
        #np.genfromtxt(Arr, delimiter=",")
        np.savetxt(pathOUT, Arr, delimiter=",")
    return Arr

def CONVERT_NAME(pathIN, nameOUT):
    Img = PIL.Image.open(pathIN)
    os.remove(pathIN)
    cv.imwrite(nameOUT, Img)
    
def CONVERT_CROP(Arr, N, M):
    print("     Crop in ", N,"x",M)
    MATRIX = []
    Matrix = np.array(Arr)
    
    W,H = Matrix.shape
    w,h = math.floor(W/N),math.floor(H/M)
    print("    ",W,"x",H,"to",w,"x",h)
    n = 0
    while n < N:
        m =0
        while m < M:
            size = [n*w,(n+1)*w, m*h,(m+1)*h]
            #print(size)
            matrix = Matrix[size[0]:size[1] , size[2]:size[3]]
            #print(matrix.shape)
            MATRIX.append(matrix)
            m += 1
        n += 1
        
    #print(Matrix.shape,"to", matrix.shape)
    return MATRIX
    
def CONVERT_CROP2(ar):
    ar_split = np.array_split(ar,5,axis=1)
    top = 0
    bot = 5
    ar = [[],[],[],[],[],[],[],[],[],[]]
    for j in range(0,5):
        ar_topbot = ar_split[j]
        ar_topbot = np.array_split(ar_topbot,2,axis=0)
        ar_top = ar_topbot[0].astype("uint16")
        ar_bot = ar_topbot[1].astype("uint16")
        ar_top = ID_renamer(ar_top)
        ar_bot = ID_renamer(ar_bot)
        ar[top] = ar_top
        ar[bot] = ar_bot
        top +=1
        bot +=1
    #print(ar)
    return ar
    
def ID_renamer(ar):
    ID = 0
    for j in np.unique(ar):
        if j !=0:
            ID +=1
            ar[ar==j] = ID
    return ar

def PLOT(Data, Algo,Title,Labels,Range,Save):
    width = 0.15
    gap = 0.2
    index = -width*len(Algo)/2
    
    x = np.arange(len(Labels))  # the label locations
    fig, ax = plt.subplots()
    
    a = 0
    X = x + index
    while a < len(Algo): #[a][metric][min,AVG,max]
        if a == 0: X += gap
        else: X += width
        
        ax.bar(     X, Data[a][1], width=width, label=Algo[a])
        ax.errorbar(X, Data[a][1], yerr=[Data[a][0], Data[a][2]], fmt='ko', capsize=5)
        a += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.title(Title)
    
    ax.set_ylabel('Index')
    ax.set_xticks(x)
    ax.set_xticklabels(Labels)
    
    ax.legend()
    fig.tight_layout()
    plt.ylim(Range[0],Range[1])
    
    if "Plot" in Save:
        plt.savefig(Title)
        
def DELTA(matrix,matrix2):
    numero=findhighestIDnumber(matrix)
    print(numero)
    numero2=findhighestIDnumber(matrix2)
    identified=0
    misidentified=0
    number=findlowestIDnumber(matrix)
    print(numero,numero2)
    numero=numero+1
    exist=existingFiberIds(matrix,numero)
    exist2 = existingFiberIds(matrix2, numero2)
    numberoffibers2=len(exist)
    numberoffibers22 = len(exist2)
    notdetected = 0
    mp = {k: 0 for k in range(int(numero2) + 5)}
    for number in exist:#exist

        gtpixel=GTpixels(matrix,matrix2,number)
        a=majorityInMatrix(gtpixel, numero2)
        if a!=0:
            mp[a] += 1
    mui = 0
    mp2 = {k: 0 for k in range(int(numero2) + 5)}
    for i in range(int(numero2)+5):
        if mp[i]>1:
            mui=mui+1
            mp2[mp[i]]=mp2[mp[i]]+1
    array=[]
    for i in range(int(numero2)+5):
        if mp2[i]>0:
            array.append((mp2[i],"groups of", i))
    delta = mui / numberoffibers2
    return delta

def findhighestIDnumber(matrix): ###largest id number in the matrix
    numero=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]>numero: numero=matrix[i][j]
    return numero

def findlowestIDnumber(matrix): ### lowest id number in the matrix
    numero= findhighestIDnumber(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]!=0:
                if matrix[i][j]<numero: numero=matrix[i][j]
    return numero

def existingFiberIds(arr, greaterthanany):  #####Checking what Fiber IDs are actually existing, returns array of the ids
    mp = {k: 0 for k in range(int(greaterthanany)+5)}
    #print(greaterthanany)
    existingfibers=[]
    #print(len(arr), len(arr[0]),"pog")
    #print(greaterthanany,"bruh")
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            #print(i,j,"ij")
            if arr[i][j]!=0:
                #print(arr[i][j], i,j)
                mp[arr[i][j]] += 1

    for i in range(int(greaterthanany)+1):
        if mp[i]!=0:
            existingfibers=np.append(existingfibers, i, axis=None)

    return existingfibers

def GTpixels(matrix,matrix2,number): #Finds the corresponding id number in the other matrix (the ids do not directly correspond)

    imax = 0
    jmax = 0
    z=0
    imin = len(matrix)
    jmin = len(matrix[0])

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == number:
                if i > imax: imax = i
                if i < imin: imin = i
                if j> jmax: jmax = j
                if j < jmin: jmin = j
    jmax=jmax+1
    imax=imax+1
    #print(zi,zj,imax, imin, jmax, jmin,number)
    #print("pixel boundary:",imin,imax,jmin,jmax)
    newmatrix=np.zeros((imax-imin,jmax-jmin),dtype="int")
    newmatrix2 = np.zeros((imax - imin, jmax - jmin), dtype="int")
    #print(newmatrix)
    for i in range(imin,imax):
        for j in range(jmin,jmax):
            if matrix[i][j]==number:
                newmatrix[i-imin][j-jmin]=matrix[i][j]
            #if matrix[i][j] ==number:
                newmatrix2[i-imin][j-jmin]=matrix2[i][j]
    #print(newmatrix,"\n", newmatrix2)
    return newmatrix2#, newmatrix3

def majorityInMatrix(arr,greaterthanany):  ###### Returns the number that happens the most times in a matrix
    mp = {k: 0 for k in range(int(greaterthanany)+1)}
    maxi=0
    maxi2=0
    #print(greaterthanany,"bruh")
    #print(arr.shape, greaterthanany)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j]!=0:
                #print(arr[i][j],"arr[i][j]")
                #print(mp[arr[i][j]])
                mp[int(arr[i][j])] += 1

                if mp[int(arr[i][j])]>maxi:
                    maxi=mp[int(arr[i][j])]
                    maxi2=arr[i][j]

    return maxi2
def existingFiberIds(arr, greaterthanany):  #####Checking what Fiber IDs are actually existing, returns array of the ids
    mp = {k: 0 for k in range(int(greaterthanany)+5)}
    #print(greaterthanany)
    existingfibers=[]
    #print(len(arr), len(arr[0]),"pog")
    #print(greaterthanany,"bruh")
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            #print(i,j,"ij")
            if arr[i][j]!=0:
                #print(arr[i][j], i,j)
                mp[arr[i][j]] += 1

    for i in range(int(greaterthanany)+1):
        if mp[i]!=0:
            existingfibers=np.append(existingfibers, i, axis=None)

    return existingfibers

# Function to find count of all
# majority elements in a Matrix
def majorityInMatrix(arr,greaterthanany):  ###### Returns the number that happens the most times in a matrix
    mp = {k: 0 for k in range(int(greaterthanany)+1)}
    maxi=0
    maxi2=0
    #print(greaterthanany,"bruh")
    #print(arr.shape, greaterthanany)
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            if arr[i][j]!=0:
                #print(arr[i][j],"arr[i][j]")
                #print(mp[arr[i][j]])
                mp[int(arr[i][j])] += 1

                if mp[int(arr[i][j])]>maxi:
                    maxi=mp[int(arr[i][j])]
                    maxi2=arr[i][j]

    return maxi2

def GTpixels(matrix,matrix2,number): #Finds the corresponding id number in the other matrix (the ids do not directly correspond)

    imax = 0
    jmax = 0
    z=0
    imin = len(matrix)
    jmin = len(matrix[0])

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == number:
                if i > imax: imax = i
                if i < imin: imin = i
                if j> jmax: jmax = j
                if j < jmin: jmin = j
    jmax=jmax+1
    imax=imax+1
    #print(zi,zj,imax, imin, jmax, jmin,number)
    #print("pixel boundary:",imin,imax,jmin,jmax)
    newmatrix=np.zeros((imax-imin,jmax-jmin),dtype="int")
    newmatrix2 = np.zeros((imax - imin, jmax - jmin), dtype="int")
    #print(newmatrix)
    for i in range(imin,imax):
        for j in range(jmin,jmax):
            if matrix[i][j]==number:
                newmatrix[i-imin][j-jmin]=matrix[i][j]
            #if matrix[i][j] ==number:
                newmatrix2[i-imin][j-jmin]=matrix2[i][j]
    #print(newmatrix,"\n", newmatrix2)
    return newmatrix2#, newmatrix3

def CountPixels(matrix,number):    #nuber of pixels with a specidic id
    pixelnumber=0

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == number:
                pixelnumber=pixelnumber+1
    return pixelnumber

def findhighestIDnumber(matrix): ###largest id number in the matrix
    numero=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]>numero: numero=matrix[i][j]
    return numero
def findlowestIDnumber(matrix): ### lowest id number in the matrix
    numero= findhighestIDnumber(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]!=0:
                if matrix[i][j]<numero: numero=matrix[i][j]
    return numero
def findlowestandhighest(matrix):
    numero2 = 1000000
    numero=0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                if matrix[i][j] < numero2: numero2 = matrix[i][j]
                if matrix[i][j] > numero: numero = matrix[i][j]
    return numero2, numero


def ComparatorOf2(matrix,matrix2): ###compares 2 matrices, returnes the number of identified and misidentified fibers
    numero=findhighestIDnumber(matrix)
    print(numero)
    numero2=findhighestIDnumber(matrix2)
    identified=0
    misidentified=0
    number=findlowestIDnumber(matrix)
    print(numero,numero2)
    numero=numero+1
    exist=existingFiberIds(matrix,numero)
    exist2 = existingFiberIds(matrix2, numero2)
    numberoffibers2=len(exist)
    numberoffibers22 = len(exist2)
    notdetected = 0
    mp = {k: 0 for k in range(int(numero2) + 5)}
    for number in exist:#exist

        gtpixel=GTpixels(matrix,matrix2,number)
        a=majorityInMatrix(gtpixel, numero2)
        if a!=0:

            pixelb=CountPixels(matrix2,a) ##########number of pixels of ID2 in the compared matrix
            pixela=CountPixels(gtpixel,a) ##########number of pixels of ID2 in the original fiber
            pixelc=CountPixels(matrix, number) ##########number of pixels of ID1 in the original fiber
            mp[a]+=1
            ratio=pixela/pixelb
            ratio2=pixela/pixelc
            print(pixela, pixelb,pixelc, ratio, ratio2, number)
            if ratio>0.8:
                identified=identified+1
            else:
                misidentified=misidentified+1
        else:
            misidentified=misidentified+1
            notdetected=notdetected+1
    mui=0
    mp2 = {k: 0 for k in range(int(numero2) + 5)}
    for i in range(int(numero2)+5):
        if mp[i]>1:
            mui=mui+1
            mp2[mp[i]]=mp2[mp[i]]+1
    array=[]
    for i in range(int(numero2)+5):
        if mp2[i]>0:
            array.append((mp2[i],"groups of", i))
    alpha=identified/numberoffibers2
    beta=misidentified/(misidentified+identified)
    gamma=notdetected/numberoffibers2
    delta=mui/numberoffibers2
    return identified, misidentified, numero, numberoffibers2, numberoffibers22, notdetected, alpha, beta, gamma, delta,array

def ComparatorM(matrix,matrix2):

    ###### Delete only if watershed
    #matrix2 = np.delete(matrix2, (0), axis=0)
    #matrix2 = np.delete(matrix2,(0), axis=1)
    #matrix = np.delete(matrix, (0), axis=0)
    #matrix = np.delete(matrix,(0), axis=1)
    print(matrix.shape)
    matrix2 = np.pad(matrix2, ((0, 2), (0, 2)))
    #matrix = np.pad(matrix, ((0, 2), (0, 2)))
    matrix = matrix.astype(int)
    matrix2= matrix2.astype(int)


    start = time.time()
    comparatorW=ComparatorOf2(matrix,matrix2)
    print("For Watershed: identified", comparatorW[0],"misidentified",comparatorW[1],"Number of fibers in GT", comparatorW[3],"Number of fibers detected by watershed ",comparatorW[4], "notdetected", comparatorW[5] )
    print("alpha",comparatorW[6],"beta",comparatorW[7] , "gamma", comparatorW[8], "delta",comparatorW[9])
    print(comparatorW[10])
#plotting = Plotting()
#generateResultsChart([0.6,0.7,0.8,0.4],[0.6,0.5,0.8,0.6],[0.6,0.6,0.8,0.4],[0.9,0.9,0.9,0.4])
#plotting.generateResultsChart
    end = time.time()
    print("Time",end - start)

#comparatorS=ComparatorOf2(matrix,matrix3)
#print(" For Stardist: identified", comparatorS[0],"misidentified",comparatorS[1])
def DELTA(matrix,matrix2):
    print(matrix.shape)
    matrix2 = np.pad(matrix2, ((0, 2), (0, 2)))
    #matrix = np.pad(matrix, ((0, 2), (0, 2)))
    #matrix = matrix.astype(int)
    #matrix2 = matrix2.astype(int)
    numero=findhighestIDnumber(matrix)
    print(numero)
    numero2=findhighestIDnumber(matrix2)
    identified=0
    misidentified=0
    number=findlowestIDnumber(matrix)
    print(numero,numero2)
    numero=numero+1
    exist=existingFiberIds(matrix,numero)
    exist2 = existingFiberIds(matrix2, numero2)
    numberoffibers2=len(exist)
    numberoffibers22 = len(exist2)
    notdetected = 0
    mp = {k: 0 for k in range(int(numero2) + 5)}
    for number in exist:#exist

        gtpixel=GTpixels(matrix,matrix2,number)
        a=majorityInMatrix(gtpixel, numero2)
        if a!=0:
            mp[a] += 1
    mui = 0
    mp2 = {k: 0 for k in range(int(numero2) + 5)}
    for i in range(int(numero2)+5):
        if mp[i]>1:
            mui=mui+1
            mp2[mp[i]]=mp2[mp[i]]+1
    array=[]
    for i in range(int(numero2)+5):
        if mp2[i]>0:
            array.append((mp2[i],"groups of", i))
    delta = mui / numberoffibers2
    print("delta",delta)
    return delta