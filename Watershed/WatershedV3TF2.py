import cv2 as cv
import imutils
import math
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import os
import pandas as pd
import statistics as stat
# from tqdm import tqdm
# import sys
import warnings
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

warnings.filterwarnings('ignore')
import time


# complete, comprehensive version (3000-10,000ms)
def WATERSHED(FileIN, R=5, RE=[2/3, 2.5, 0.85], PMSF=[5, 5, 6], ke=3, SSp=200):
                     #F_mean   Dist FibR Ecc.         IT, 1  2         Substeps
    T0 = time.time()

    # FUNCTIONS 
    def COL(x, y, z, ID):
        X = int((x / width) * 255)
        Y = int((y / height) * 255)
        Z = int(z ** 2)
        Col = (X, Y, Z)
        ID += 1
        return Col, ID

    # Quality of Life
    def PRINT(text):
        print("\n")
        print(text)

    def PROGRESS(iteration, total, prefix='', suffix='', decimals=0, length=10, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration) / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r  {prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print('\n')

    # PARAMETERS:
    PRINT("INIT :")
    T1 = time.time()
    # System:
    input_file = FileIN  # name, filetype
    output_file = [input_file[0], ".png", ".csv"]
    path_R_input = "../Data"
    path_R_output = "../Data Processed/Watershed"

    # Watershed
    F_mean = R
    D_RE, F_RE, E_RE = RE
    R_min = F_mean * (1 - D_RE)
    MinDist = math.floor(R_min)

    # PMSF - iterations,spatial radius,color radius
    PyrFiltIT, PyrFilt1, PyrFilt2 = PMSF

    # Extra Processing
    if ke < 0: ke = 0  # kernel (0==none)
    FitEllipse = True

    # Display
    Show_In = True
    Show_PyrFilt = True
    Show_Otsu = True
    Show_Boundary = True
    Show_Shapes = True
    Show_Fibers = True
    Show_Output = True

    Show_Fitted = True
    Show_Center = True

    Col_ShapeCenter = (255, 255, 255)
    Col_Boundary = (0, 255, 0)
    Col_Background = (100, 100, 100)

    Print_Matrix = False
    Print_Output = False

    print('INITIAL PARAMETERS')
    print("[INFO] mean fiber radius     :", format(round(F_mean, 3)))
    print("[INFO] distance T. factor    :", format(round(D_RE * 100)), "%")
    print("[INFO] radius error factor   :", format(round(F_RE, 3)))
    if FitEllipse:
        print("[INFO] fiber ecc. error      :", format(round(E_RE, 3)))
    print("[INFO] PMS filter            :", PyrFiltIT, PyrFilt1, PyrFilt2)
    print("[INFO] Noise Kernel Radius   :", format(ke))

    print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")

    # IMAGE PROCESSING
    PRINT("IMAGE PROCESSING : ")
    T2 = time.time()
    # Open image
    print("Open image ...")
    path_script = os.path.dirname(__file__)
    path = os.path.join(path_script, path_R_input, (input_file[0] + input_file[1]))
    img = cv.imread(path)
    print(path)
    if Show_In:
        cv.imshow('Input', img)
        cv.waitKey(1)
    height, width, _ = img.shape

    # Smoothing / de-noising
    print("Pyramid Mean Shift Filter ...")  # replace 1 by PytFilt1 <> progress bar
    p = PyrFiltIT
    
    SSp_0 = time.time()
    imgPMSF = img
    while p > 0:
        p -= 1
        PROGRESS(PyrFiltIT - p, PyrFiltIT, prefix='', suffix='', length=30)
        imgPMSF = cv.pyrMeanShiftFiltering(imgPMSF, PyrFilt1, PyrFilt2)
        
        SSp_1  =time.time()
        D_SSp = (SSp_1 - SSp_0)*1000
        if Show_PyrFilt and (D_SSp >= SSp):
            SSp_0 = SSp_1
            cv.imshow('imagePMSF', imgPMSF)
            cv.waitKey(1)

    # Otsu
    print("Otsu binarization ...")
    gray = cv.cvtColor(imgPMSF, cv.COLOR_BGR2GRAY)
    imgTSH = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    # erode-dilate
    print("Noise Reduction...")
    if ke != 0:
        kernel = np.ones((ke, ke), np.uint8)
        imgTSH = cv.dilate(cv.erode(imgTSH, kernel), kernel)
    if Show_Otsu:
        cv.imshow("Thresh", imgTSH)
        cv.waitKey(1)

    print("> " + str(round((T2 - T1) * 1000)) + "[ms] <")

    # WATERSHED
    PRINT("WATERSHED : ")
    T3 = time.time()
    # labelling
    print("Labelling ...")
    D = ndimage.distance_transform_edt(imgTSH)
    localMax = peak_local_max(D, indices=False, min_distance=MinDist, labels=imgTSH)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=imgTSH)

    # setup
    R = A = B = set([])  # shapes stats (radius)
    Shapes = []
    arr_out = np.zeros((height, width), dtype=int)
    img_out = np.zeros((height, width, 3), np.uint8)
    img_out[:] = Col_Background
    img_B = cv.imread(path)
    img_S = cv.imread(path)
    CX = 0

    # identification
    print("  mask > label > find contours > shape fitting")
    if FitEllipse:
        print("  Shape == Ellipse")
    else:
        print("  Shape == Circle")
    Progress = len(np.unique(labels))
    progress = 1
    SSp_0  = time.time()
    
    for label in np.unique(labels):
        progress += 1
        PROGRESS(progress, Progress, prefix='', suffix='', length=30)
        # eliminate background
        if label == 0:
            continue

        # mask and label region
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # find contours
        Cnts = cv.findContours(mask.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(Cnts)
        c = max(cnts, key=cv.contourArea)
        if Show_Boundary:
            cntsX = cv.findContours(imgTSH.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cntsX = imutils.grab_contours(cntsX)
            for (i, cX) in enumerate(cnts):
                CX += 1
                cv.drawContours(img_B, [cX], -1, Col_Boundary, 1)

        # SHAPE FITTING
        # draw enclosing circle 
        ID = 1
        if not FitEllipse:
            ((x, y), r) = cv.minEnclosingCircle(c)
            Col, ID = COL(x, y, r, ID)
            Shapes.append([x, y, r])
            if Show_Shapes and Show_Fitted and not FitEllipse:
                cv.circle(img_S, (int(x), int(y)), int(r), Col, 1)
            Shapes.append([x, y, r])
            R.add(r)
        # draw inscribed-bound ellipse
        if FitEllipse:
            # bounding box
            rect = (x, y), (w, h), Agl = cv.minAreaRect(c)
            Col, ID = COL(x, y, Agl, ID)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            # ellipse
            # (x,y),(a,b),Agl = cv.fitEllipse(box)
            if Show_Shapes and Show_Fitted:
                img_S = cv.ellipse(img_S, ((x, y), (w, h), Agl), Col, 1)
                img_S = cv.drawContours(img_S, [box], 0, Col, 1)
                # compute eccentricty
            if w >= h:
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

            # draw center
        if Show_Center:
            cv.circle(img_S, (int(x), int(y)), 1, Col_ShapeCenter, -1)

        if (Show_Boundary or Show_Shapes):
            SSp_1 = time.time()
            D_SSp = (SSp_1 - SSp_0)*1000
            if (D_SSp > SSp or progress>=Progress):
                SSp_0 = SSp_1
                if Show_Boundary: cv.imshow("Boundaries", img_B)
                if Show_Shapes: cv.imshow("Shapes", img_S)
                cv.waitKey(1)

    print("> " + str(round((T3 - T2) * 1000)) + "[ms] <")

    PRINT("FIBER IDENTIFICATION :")
    T4 = time.time()
    # Discriminator
    print("Defining discriminator ...")
    if not FitEllipse:
        R_med = stat.median(R)
        R_avg = stat.mean(R)
    else:
        A_med = stat.median(A)
        A_avg = stat.mean(A)
        B_med = stat.median(B)
        B_avg = stat.mean(B)

    F = set([])  # same but filtered
    Fibers = []
    img_F = cv.imread(path)

    ID = 1

    print("sizing > test > drawing and appending")
    Progress = len(Shapes)
    progress = 1
    SSp_0 = time.time()
    for shape in Shapes:
        PROGRESS(progress, Progress, prefix='', suffix='', length=30)
        progress += 1
        # fit enclosing circle
        x, y = shape[0], shape[1]
        if not FitEllipse:
            r = shape[2]
            if R_med / F_RE < r < R_med * F_RE:
                F.add(r)
                Col, ID = COL(x, y, r, ID)
                cv.circle(arr_out, (int(x), int(y)), int(r), ID, -1)
                if Show_Fitted:
                    cv.circle(img_F, (int(x), int(y)), int(r), Col, -1)
                    cv.circle(img_out, (int(x), int(y)), int(r), Col, -1)
                if Show_Center:
                    cv.circle(img_F, (int(x), int(y)), 1, Col_ShapeCenter, -1)
                # fiber ID and appending
                Fibers.append([len(F), round(x), round(y), round(r, 3)])
        if FitEllipse:
            a, b, Agl, c, e = shape[2:]
            if abs(e) < E_RE and B_med / F_RE < min(a, b) and max(a, b) < A_med * F_RE:
                F.add(c)
                Col, ID = COL(x, y, c, ID)
                cv.ellipse(arr_out, ((x, y), (a, b), Agl), ID, -1)
                if Show_Fitted:
                    cv.ellipse(img_F, ((x, y), (a, b), Agl), Col, -1)
                    cv.ellipse(img_out, ((x, y), (a, b), Agl), Col, -1)
                if Show_Center:
                    cv.circle(img_F, (int(x), int(y)), 1, Col_ShapeCenter, -1)
                # fiber ID and appending
                Fibers.append([len(F), round(x), round(y), round(c, 3), round(e, 3)])

        SSp_1 = time.time()
        D_SSp = (SSp_1 - SSp_0)*1000
        if (D_SSp > SSp or progress>=Progress):
            SSp_0 = SSp_1
            if Show_Fibers: cv.imshow("Fibers", img_F)
            if Show_Output: cv.imshow("OUTPUT", img_out)
            cv.waitKey(1)

    print("> " + str(round((T4 - T3) * 1000)) + "[ms] <")

    # OUTPUT
    PRINT("OUTPUT")
    print("statisctics ...")
    # statistics:

    if FitEllipse:
        E = []
        for ellipse in Fibers:
            E.append(ellipse[-1])
        E_avg = stat.mean(E)
        E_med = stat.median(E)

    # save:
    if Print_Output or Print_Matrix: print("save results ...")
    if Print_Output:  # ID'ed Fibers only image
        # save to .png
        print("\n")
        print('image to file :')
        path_script = os.path.dirname(__file__)
        path = os.path.join(path_script, path_R_output)
        os.chdir(path)
        print(path)
        filename = (output_file[0] + output_file[1])
        print(filename)
        cv.imwrite(filename, img_out)
        print('Successfully saved')

    if Print_Matrix:  # ID'ed Fibers only matrix
        # delete first col & row
        arr_out = np.delete(arr_out, 0, 0)
        arr_out = np.delete(arr_out, 0, 1)
        # save to .csv
        print("\n")
        print('matrix to file :')
        path_script = os.path.dirname(__file__)
        path = os.path.join(path_script, path_R_output)
        os.chdir(path)
        path = path + "\ " + input_file[0] + output_file[2]
        print(path)
        # numpy.savetxt((outpit_file[0]+output_file[2]),a,delimiter="")
        pd.DataFrame(arr_out).to_csv((path), header="none", index="none")
        print('Successfully saved')

    # print stats
    print('WATERSHED')
    print("[INFO] unique contours found :", format(CX))
    print("[INFO] unique shapes found   :", format(max(len(R), len(A))))
    print("[INFO] unique fibers found   :", format(len(F)))
    print("\n\n -----")
    print('MEAN VALUES')
    if FitEllipse:
        print("[INFO] semi-major axis   :", format(round(A_avg / 2, 3)))
        print("[INFO] semi-minor axis   :", format(round(B_avg / 2, 3)))
        print("[INFO] eccentricity      :", format(round(E_avg, 3)))
    else:
        print("[INFO] radius    :", format(R_avg))
    print("\n\n -----")
    print('MEDIAN VALUES')
    if FitEllipse:
        print("[INFO] semi-major axis   :", format(round(A_med / 2, 3)))
        print("[INFO] semi-minor axis   :", format(round(B_med / 2, 3)))
        print("[INFO] eccentricity      :", format(round(E_med, 3)))
    else:
        print("[INFO] radius   :", format(R_med))

    T5 = time.time()
    print("> " + str(round((T5 - T4) * 1000)) + "[ms] <")
    print("\n\n -----")
    print("> " + str(round((T5 - T0) * 1000)) + "[ms] <")

    return arr_out,T_0

print("----- START PROGRAM ----- \n")
T_00 = time.time()
Dir = "Tape_B/Images/"
Type=".jpg"
Name = "Tape_B"
N=n=2
M=m=20
I = 0
while m > 1:
    print("\n\n\n ----- STARTFILE -----")
    I+=1
    print("Number :" + str(I))
    name = Name+"_"+str(n)+"_"+str(m)
    print(str(name))
    path = Dir+name+Type
    print(str(path))
    m -= 1
    input_file = [Dir+name, ".jpg"]
    OUTPUT,T_0 = WATERSHED(input_file)  # (Name, Filetype)
    T_6 = time.time()
    print("> " + str(round((T_6 - T_0)*1000)) + "[s] <")
    print("----- ENDFILE -----\n\n\n")
T_11 = time.time()
print("----- END PROGRAM ----- \n")
print("> " + str(round((T_11 - T_00),1)) + "[s] <")

cv.waitKey(0)