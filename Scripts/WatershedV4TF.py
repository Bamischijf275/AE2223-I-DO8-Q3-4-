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


# V4=barebones, fast version (900-1200ms)
def WATERSHED(FileIN, R=5, RE=[2 / 3, 2, 0.85], PMSF=[10, 4, 5], ke=3):
    T0 = time.time()

    # FUNCTIONS 
    def COL(x, y, z, ID):
        X = int((x / width) * 255)
        Y = int((y / height) * 255)
        Z = int(z ** 2)
        Col = (X, Y, Z)
        ID += 1
        return Col, ID

    # PARAMETERS:
    # System:
    input_file = [FileIN[0]+FileIN[1],FileIN[2]]  # name, filetype
    output_file = [FileIN[1], ".png", ".csv"]
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

    Show_Fitted = True

    Col_Background = (100, 100, 100)

    Print_Matrix = True
    Print_Output = False

    # IMAGE PROCESSING
    # Open image
    path_script = os.path.dirname(__file__)
    path = os.path.join(path_script, path_R_input, (input_file[0] + input_file[1]))
    img = cv.imread(path)
    height, width, _ = img.shape

    # Smoothing / de-noising
    imgPMSF = img
    while PyrFiltIT > 0:
        PyrFiltIT -= 1
        imgPMSF = cv.pyrMeanShiftFiltering(imgPMSF, PyrFilt1, PyrFilt2)

    # Otsu
    gray = cv.cvtColor(imgPMSF, cv.COLOR_BGR2GRAY)
    imgTSH = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    # erode-dilate
    if ke != 0:
        kernel = np.ones((ke, ke), np.uint8)
        imgTSH = cv.dilate(cv.erode(imgTSH, kernel), kernel)

    # WATERSHED
    # labelling
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

    # identification
    for label in np.unique(labels):
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

        # SHAPE FITTING
        # draw enclosing circle 
        ID = 1
        if not FitEllipse:
            ((x, y), r) = cv.minEnclosingCircle(c)
            Col, ID = COL(x, y, r, ID)
            Shapes.append([x, y, r])
            Shapes.append([x, y, r])
            R.add(r)
        # draw inscribed-bound ellipse
        if FitEllipse:
            # bounding box
            rect = (x, y), (w, h), Agl = cv.minAreaRect(c)
            Col, ID = COL(x, y, Agl, ID)
            box = cv.boxPoints(rect)
            box = np.int0(box)
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

            # Discriminator
    if not FitEllipse:
        R_med = stat.median(R)
    else:
        A_med = stat.median(A)
        B_med = stat.median(B)

    F = set([])  # same but filtered
    Fibers = []
    img_F = cv.imread(path)

    ID = 1

    for shape in Shapes:
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
                # fiber ID and appending
                Fibers.append([len(F), round(x), round(y), round(c, 3), round(e, 3)])

    # OUTPUT
    # save:
    if Print_Output:  # ID'ed Fibers only image
        # save to .png
        path_script = os.path.dirname(__file__)
        path = os.path.join(path_script, path_R_output)
        os.chdir(path)
        filename = (output_file[0] + output_file[1])
        cv.imwrite(filename, img_out)

    if Print_Matrix:  # ID'ed Fibers only matrix
        # save to .csv
        path_script = os.path.dirname(__file__)
        path = os.path.join(path_script, path_R_output)
        os.chdir(path)
        path = path + "\ " + input_file[0] + output_file[2]
        pd.DataFrame(arr_out).to_csv((path), header="none", index="none")

    return arr_out, T0

print("----- START PROGRAM ----- \n")
T00 = time.time()
Dir = "Tape_B/Images/"
Name = "Tape_B"
Type=".jpg.tif"
M=10
N=[1,2,3,4,6,8,9,10]
n=0
m = 1
while n+1 < len(N):
    while m <= M:
        print("\n\n ----- STARTFILE -----")
        print("Number :" + str(N[n]) + "_" + str(m))
        name = Name+"_"+str(N[n])+"_"+str(m)
        print(str(name))
        path = Dir+name+Type
        print(str(path))
        input_file = [Dir, name, Type]
        OUTPUT,T0 = WATERSHED(input_file)  # (Name, Filetype)
        T6 = time.time()
        print("> " + str(round((T6 - T0)*1000)) + "[s] <")
        print("------ ENDFILE ------")
        m+=1
    n+=1
T11 = time.time()
print("----- END PROGRAM ----- \n")
print("> " + str(round((T11 - T00),1)) + "[s] <")

cv.waitKey(0)
