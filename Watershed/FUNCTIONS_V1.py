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
    # parameters
    if "print" in DETAIL[0]:
        print("\n init :")
        T0 = time.time()

    Col_CM = [(0, 255, 0), (150, 150, 150), (0, 0, 255), (255, 0, 0, 0)]  # TP-G,TN-,FP-B,FN-R
    Col_Background = (100, 100, 100)

    Cutoff = PARAMETERS[0]
    ShowTime = DETAIL[1] / 1000
    extra_IMGS = []

    Result = [[0, 0, 0, 0], [0, 0, 0, 0], 0, 0, 0]  # TP,TN,FP,FN (pixels, fibers), TrueFib, ResultFib, True Area

    # format matrices
    SizeT = MatrixT.shape
    SizeR = MatrixR.shape
    if "print" in DETAIL[0]:
        print("     Matrix Size T,R Input:", SizeT, SizeR)

    # MatrixT = MatrixT.astype(int)
    MatrixT = np.delete(MatrixT, (0), axis=0)
    MatrixT = np.delete(MatrixT, (0), axis=1)
    SizeT = MatrixT.shape

    # MatrixR = MatrixR.astype(int)
    MatrixR = np.delete(MatrixR, (0), axis=0)
    MatrixR = np.delete(MatrixR, (0), axis=1)
    SizeR = MatrixR.shape

    if SizeT != SizeR:
        Matrix = np.zeros(SizeT)
        Matrix[:MatrixR.shape[0], :MatrixR.shape[1]]
        MatrixR = Matrix
    if SizeT != SizeR:
        Matrix = np.zeros(SizeR)
        Matrix[:MatrixT.shape[0], :MatrixT.shape[1]]
        MatrixT = Matrix
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
    FibersT = MatrixID(MatrixT)
    FibersR = MatrixID(MatrixR)

    Result[2] += len(FibersT)
    Result[3] += len(FibersR)
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
            if MatrixCount(SubMatrixR, ID) >= Nmax:
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

        h, w = SubMatrixT.shape
        Area = h * w
        TP = MatrixCount(MUL_Matrix, 1)
        FN = MatrixCount(DIF_Matrix, 1)
        FP = MatrixCount(DIF_Matrix, -1)
        TN = Area - TP - FP - FN

        # Results
        Result[0][0] += TP
        Result[0][1] += TN
        Result[0][2] += FP
        Result[0][3] += FN

        if (TP + FP) == 0:  # none found - FN
            Result[1][3] += 1
        elif TP / (FP + TP) >= Cutoff:  # TP
            Result[1][0] += 1

        elif FP == 0 or FN == 0:  # no overlap - no change
            Result = Result
        elif FP >= FN:  # FP
            Result[1][2] += 1
        elif FN > FP:  # FN
            Result[1][3] += 1

        else:
            Result[1][1] += 1
            ID_p = ID_T

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
            print(TP, TN, FP, FN)

    if "draw" in DETAIL[0]:
        cv.imshow("Accuracy", imgConf)
        cv.waitKey(1)

    # format results
    Result[4] = np.count_nonzero(MatrixT)
    Result[0][1] = np.count_nonzero(MatrixT==0)
    
    if "print" in DETAIL[0]:
        T1 = time.time()
        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")
    return (Result, extra_IMGS)


# others

# WT

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
                name = Name + "_2_-" + str(N[i])
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
                name = Name + "_2_-" + str(rnd.randint(N[0], N[1]))
            else:
                name = Name + "_" + str(rnd.randint(N[0], N[1])) + "_" + str(rnd.randint(M[0], M[1]))
            Names.append(name)
            i += 1
        
    else:
        if N == [] and M==[]: # default
            n, m = 2,1
        else:
            n,m = N[0],M[0]
            
        if tape == "Large" or tape == "Cropped":
            name + "_2_-" + str(n)
        else:
            name = Name + "_" + str(n) + "_" + str(m)
        Names.append(name)
        
    return Names


def CONVERT_TIFtoCSV(pathIN, pathOUT):
    Img = PIL.Image.open(pathIN)
    Arr = np.array(Img)
    if pathOUT != "":
        pd.DataFrame(Arr).to_csv((pathOUT), header="none", index="none")
    # Arr_csv = np.genfromtxt(Arr, delimiter=",")
    return Arr

def CONVERT_NAME(pathIN, nameOUT):
    Img = PIL.Image.open(pathIN)
    os.remove(pathIN)
    cv.imwrite(nameOUT, Img)
    
def CONVERT_CROP(Arr, M, N):
    print("     Crop in ", M,"x",N)
    MATRIX = []
    Matrix = np.array(Arr)
    
    H,W = Matrix.shape
    w,h = math.floor(W/M),math.floor(H/N)
    print("    ",W,"x",H,"to",w,"x",h)
    
    m = 0
    while m < M:
        n =0
        while n < N:
            matrix = MATRIX[m*h:(m+1)*h][n*w:(n+1)*w]
            MATRIX.append(matrix)
            n += 1
        m += 1
    return MATRIX
    
    