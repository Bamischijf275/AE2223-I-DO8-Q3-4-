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
import sys
import time
import warnings
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)


# complete, comprehensive version (3000-10,000ms)
def WATERSHED(FILE, PROGRAM, PARAMETERS):
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
        # Print Progress bar
        percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration) / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r  {prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration >= total:
            print('\n')

    # PARAMETERS:
    # file
    F_name = FILE[0][3]
    F_IN_path = FILE[1][0]
    F_IN_type = FILE[1][1]
    F_OUT_path = FILE[2][0]
    F_OUT_img_type = FILE[2][1][1]
    F_OUT_mat_type = FILE[2][2][1]
    
    if FILE[2][1][2] == "crop":
        F_OUT_Crop = FILE[2][1][3]
    else:
        F_OUT_Crop = [""]
        
    Print_Image = False
    Print_Matrix = False
    if FILE[2][1][0] == "save": Print_Image = True
    if FILE[2][2][0] == "save": Print_Matrix = True

    # program
    P_runtime = PROGRAM[0]
    P_SSp = PROGRAM[1]
    P_shape = PROGRAM[2]

    # parameters
    F_mean = PARAMETERS[0]
    D_RE, F_RE, E_RE = PARAMETERS[1]
    PyrFiltIT, PyrFilt1, PyrFilt2 = PARAMETERS[2]  # PMSF - iterations,spatial radius,color radius
    ke = PARAMETERS[3]

    R_min = F_mean * (1 - D_RE)
    MinDist = math.floor(R_min)

    if P_runtime == ("full" or "print"):
        PRINT("INIT :")
        T1 = time.time()

    # SETUP
    # Display
    if P_runtime == ("full" or "img"):
        Show_In = True
        Show_PyrFilt = True
        Show_Otsu = True
        Show_Boundary = True
        Show_Shapes = True
        Show_Fibers = True
        Show_Center = True
        Show_Output = True

    elif P_runtime == "fast":
        Show_In = False
        Show_PyrFilt = False
        Show_Otsu = False
        Show_Boundary = False
        Show_Shapes = False
        Show_Fibers = False
        Show_Center = False
        Show_Output = False
        Show_Fitted = False

    Col_ShapeCenter = (255, 255, 255)
    Col_Boundary = (0, 255, 0)
    Col_Background = (100, 100, 100)

    # Init Print
    if P_runtime == ("full" or "print"):
        print('INITIAL PARAMETERS')
        print("[INFO] mean fiber radius     :", format(round(F_mean, 3)))
        print("[INFO] distance T. factor    :", format(round(D_RE * 100)), "%")
        print("[INFO] radius error factor   :", format(round(F_RE, 3)))
        if P_shape == "ellipse":
            print("[INFO] fiber ecc. error      :", format(round(E_RE, 3)))
        print("[INFO] PMS filter            :", PyrFiltIT, PyrFilt1, PyrFilt2)
        print("[INFO] Noise Kernel Radius   :", format(ke))

        print("> " + str(round((T1 - T0) * 1000)) + "[ms] <")

    # IMAGE PROCESS
    # Open image
    T2 = time.time()
    if P_runtime == ("full" or "print"): print("Open image ...")

    name = F_name + F_IN_type
    path_script = os.path.dirname(__file__)
    path = os.path.join(path_script, F_IN_path, name)
    img = cv.imread(path)
    if P_runtime == ("full" or "print" or "img"):
        print(path)
        if Show_In:
            cv.imshow('Input', img)
            cv.waitKey(1)
    height, width, _ = img.shape

    # Pre-processing
    if ke < 0: ke = 0  # kernel (0==none)

    # Smoothing / de-noising
    if P_runtime == ("full" or "print"):
        print("Pyramid Mean Shift Filter ...")

    SSp_0 = time.time()
    imgPMSF = img
    p = PyrFiltIT

    while p > 0:
        p -= 1
        imgPMSF = cv.pyrMeanShiftFiltering(imgPMSF, PyrFilt1, PyrFilt2)

        if P_runtime == ("full" or "img"):
            PROGRESS(PyrFiltIT - p, PyrFiltIT, prefix='', suffix='', length=30)
            SSp_1 = time.time()
            D_SSp = (SSp_1 - SSp_0) * 1000
            if Show_PyrFilt and (D_SSp >= P_SSp or p == 1):
                SSp_0 = SSp_1
                cv.imshow('imagePMSF', imgPMSF)
                cv.waitKey(1)

    # Otsu
    if P_runtime == ("full" or "print"):
        print("Otsu binarization ...")
    gray = cv.cvtColor(imgPMSF, cv.COLOR_BGR2GRAY)
    imgTSH = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    if P_runtime == ("full" or "img") and Show_Otsu:
        cv.imshow("Thresh", imgTSH)
        cv.waitKey(1)

    # erode-dilate
    if P_runtime == ("full" or "print"):
        print("Noise Reduction...")
    if ke != 0:
        kernel = np.ones((ke, ke), np.uint8)
        imgTSH = cv.dilate(cv.erode(imgTSH, kernel), kernel)

    # WATERSHED
    if P_runtime == ("full" or "print"):
        print("> " + str(round((T2 - T1) * 1000)) + "[ms] <")
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
    Boundaries = []
    arr_out = np.zeros((height, width), dtype=int)
    img_out = np.zeros((height, width, 3), np.uint8)
    img_out[:] = Col_Background
    img_B = cv.imread(path)
    img_S = cv.imread(path)
    CX = 0

    # identification
    if P_runtime == ("full" or "print"):
        print("  mask > label > find contours > shape fitting")
        if P_shape == "ellipse":
            print("  Shape == Ellipse")
        else:
            print("  Shape == Circle")
        Progress = len(np.unique(labels))
        progress = 0
        SSp_0 = time.time()

    # Shape Fitting Loop
    for label in np.unique(labels):
        if P_runtime == "full":
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

        cntsX = cv.findContours(imgTSH.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cntsX = imutils.grab_contours(cntsX)
        for (i, cX) in enumerate(cnts):
            CX += 1
            Boundaries.append([cX])
            if P_runtime == ("full" or "img") and Show_Boundary:
                cv.drawContours(img_B, [cX], -1, Col_Boundary, 1)

        # SHAPE FITTING

        # draw enclosing circle 
        ID = 1
        if P_shape == "circle":
            ((x, y), r) = cv.minEnclosingCircle(c)
            Col, ID = COL(x, y, r, ID)
            Shapes.append([x, y, r])
            if P_runtime == ("full" or "img") and Show_Shapes:
                cv.circle(img_S, (int(x), int(y)), int(r), Col, 1)
            Shapes.append([x, y, r])
            R.add(r)

        # draw inscribed-bound ellipse
        if P_shape == "ellipse":
            # bounding box
            rect = (x, y), (w, h), Agl = cv.minAreaRect(c)
            Col, ID = COL(x, y, Agl, ID)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            if P_runtime == ("full" or "img") and Show_Shapes:
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

        # draw
        if P_runtime == ("full" or "img"):
            if Show_Center:
                cv.circle(img_S, (int(x), int(y)), 1, Col_ShapeCenter, -1)
            if Show_Boundary: cv.imshow("Boundaries", img_B)
            if Show_Shapes: cv.imshow("Shapes", img_S)

            SSp_1 = time.time()
            D_SSp = (SSp_1 - SSp_0) * 1000
            if (D_SSp > P_SSp or progress >= Progress):
                SSp_0 = SSp_1
                cv.waitKey(1)

    # FIBER SELECTION - OUTLIER
    if P_runtime == ("full" or "print"):
        print("> " + str(round((T3 - T2) * 1000)) + "[ms] <")

        print("FIBER IDENTIFICATION :")
        T4 = time.time()

        print("Defining discriminator ...")

    # setup
    if P_shape == "circle":
        R_med = stat.median(R)
        R_avg = stat.mean(R)
    elif P_shape == "ellipse":
        A_med = stat.median(A)
        A_avg = stat.mean(A)
        B_med = stat.median(B)
        B_avg = stat.mean(B)

    F = set([])  # same but filtered
    Fibers = []
    img_F = cv.imread(path)
    ID = 1

    # infos
    if P_runtime == ("full" or "print"):
        print("sizing > test > drawing and appending")
    if P_runtime == "full":
        Progress = len(Shapes)
        progress = 1
        SSp_0 = time.time()

    # Discriminant Loop
    for shape in Shapes:
        if P_runtime == "full":
            PROGRESS(progress, Progress, prefix='', suffix='', length=30)
            progress += 1

        # circle
        x, y = shape[0], shape[1]
        if P_shape == "circle":
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
        # ellipse
        if P_shape == "ellipse":
            a, b, Agl, c, e = shape[2:]
            if abs(e) < E_RE and B_med / F_RE < min(a, b) and max(a, b) < A_med * F_RE:
                F.add(c)
                Col, ID = COL(x, y, c, ID)
                cv.ellipse(arr_out, ((x, y), (a, b), Agl), ID, -1)
                if Show_Shapes:
                    cv.ellipse(img_F, ((x, y), (a, b), Agl), Col, -1)
                    cv.ellipse(img_out, ((x, y), (a, b), Agl), Col, -1)
                if Show_Center:
                    cv.circle(img_F, (int(x), int(y)), 1, Col_ShapeCenter, -1)
                # fiber ID and appending
                Fibers.append([len(F), round(x), round(y), round(c, 3), round(e, 3)])

        # info
        if P_runtime == "full":
            SSp_1 = time.time()
            D_SSp = (SSp_1 - SSp_0) * 1000
            if (D_SSp > P_SSp or progress >= Progress):
                SSp_0 = SSp_1
                if Show_Fibers: cv.imshow("Fibers", img_F)
                if Show_Output: cv.imshow("OUTPUT", img_out)
                cv.waitKey(1)

    if P_runtime == ("full" or "print"):
        print("> " + str(round((T4 - T3) * 1000)) + "[ms] <")

    # OUTPUT
    # info
    if P_runtime == ("full" or "print"):
        print("OUTPUT")
        print("statisctics ...")

        if P_shape == "ellipse":
            E = []
            for ellipse in Fibers:
                E.append(ellipse[-1])
            E_avg = stat.mean(E)
            E_med = stat.median(E)

    # save:
    if (Print_Image or Print_Matrix):
        if P_runtime == ("full" or "print"):
            print("save results ...")
        if F_OUT_Crop != [""]:
            height, width, _ = img_out.shape
            X, Y = F_OUT_Crop[1]
            n, m =height/Y, width/X
            
    if Print_Image and P_runtime ==("full"or"img"):  # ID'ed Fibers only image
        # save to .type
        print("\n")
        print('image to file :')
        name = F_name + F_OUT_img_type
        print(name)
        path_script = os.path.dirname(__file__)
        path = os.path.join(path_script, F_OUT_path)
        os.chdir(path)
        path=str(path+name)
        print(path)
        
        if F_OUT_Crop != [""]:
            #crop
            
            #save
            print("a")
            
        else:
            cv.imwrite(path, img_out)
        
        print('Successfully saved')

    if Print_Matrix:  # ID'ed Fibers only matrix
        # save to .csv
        print("\n")
        print('matrix to file :')
        name = F_name + F_OUT_mat_type
        print(name)
        path_script = os.path.dirname(__file__)
        path = os.path.join(path_script, F_OUT_path)
        os.chdir(path)
        path=str(path+name)
        print(path)
        
        if F_OUT_Crop != [""]:
            #crop
            
            #save
            print("b")
            
        else:
            pd.DataFrame(arr_out).to_csv((path), header="none", index="none")
        
        print('Successfully saved')

    # print stats
    if P_runtime == ("full" or "print"):
        print('\n')

        print('WATERSHED')
        print("[INFO] unique contours found :", format(CX))
        print("[INFO] unique shapes found   :", format(max(len(R), len(A))))
        print("[INFO] unique fibers found   :", format(len(F)))
        print("\n\n -----")

        print('MEAN VALUES')
        if P_shape == "circle":
            print("[INFO] radius    :", format(R_avg))
        elif P_shape == "ellipse":
            print("[INFO] semi-major axis   :", format(round(A_avg / 2, 3)))
            print("[INFO] semi-minor axis   :", format(round(B_avg / 2, 3)))
            print("[INFO] eccentricity      :", format(round(E_avg, 3)))
        print("\n\n -----")

        print('MEDIAN VALUES')
        if P_shape == "circle":
            print("[INFO] radius   :", format(R_med))
        elif P_shape == "ellipse":
            print("[INFO] semi-major axis   :", format(round(A_med / 2, 3)))
            print("[INFO] semi-minor axis   :", format(round(B_med / 2, 3)))
            print("[INFO] eccentricity      :", format(round(E_med, 3)))

        T5 = time.time()
        print("> " + str(round((T5 - T4) * 1000)) + "[ms] <")
        print("\n\n -----")

        T6 = time.time()
        print("> " + str(round((T6 - T0) * 1)) + "[s] <")
    return arr_out


# MAIN
print("----- START PROGRAM ----- \n")
T00 = time.time()

# SETUP
File = [
    ["Tape_B", [2, 2], [17, 37], "name"],  # File
    ["../Data/Tape_B/Tape_B_2/", ".jpg"],  # IN
    ["../Data Processed/Watershed/Training/",  # OUT
     ["save", ".jpg", "crop", [120,155]],  # Image save
     ["save", ".csv", "crop", [120,155]]  # Matrix save
     ]
]
Program = ["full",  # fast-print-img-full-wait
           250,  # Substeps
           "ellipse"  # shape=circle,ellipse,contour
           ]


Parameters = [5, [2/3, 2.5, 0.85], [7, 8, 3], 3]  # Radius, Relative errors, Filter, kernel
# LOOP
N = File[0][1]
M = File[0][2]

n = N[0]
while n <= N[1]:
    m = M[0]
    while m <= M[1]:
        print("\n ----- NEWFILE -----")
        name = File[0][0] + "_" + str(n) + "_-" + str(m)
        File[0][3] = name
        print("Image : ", name)

        Result = WATERSHED(File, Program, Parameters)
        m += 1
    n += 1
T11 = time.time()
print("\n ----- END PROGRAM ----- \n")
print("> " + str(round((T11 - T00), 1)) + "[s] <")

if Program[0] == "wait":
    cv.waitKey(0)
else: cv.waitKey(1)
