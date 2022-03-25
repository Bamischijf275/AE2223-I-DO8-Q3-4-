import cv2 as cv
import imutils
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import os
import statistics as stat
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import pandas as pd
import math

def WATERSHED (Name, Filetype, PathIN, pathOUT):

    # PARAMETER DEFINITION
    # Watershed
    F_mean = 5
    F_RE = 1 / 3  # strandard deviations
    E_RE = math.exp(-F_RE)
    R_min = F_mean * (1 - F_RE)
    MinDist = int(R_min - 1)
    
    PyrFilt1 = 14  # iterations
    PyrFilt2 = 12  # strength
    
    # Extra Processing
    ke = 3  # kernel
    FitEllipse = True
    
    #Display
    Show_In = True
    Show_Otsu = True
    Show_PyrFilt = True
    Show_Shapes = True
    Show_Fibers = True
    
    Show_Boundary = True
    Show_Fitted = True
    Show_Center = True
    Col_ShapeCenter = (255, 255, 255)
    Col_Boundary = (0, 255, 0)
    Col_FiberCircle = (0, 0, 255)
    Col_Background = (100, 100, 100)
    Col_Background = (100, 100, 100)
    Print_Matrix = True
    Print_Output = True
    
    # IMAGE PROCESSING
    # Open image
    path_script = os.path.dirname(__file__)
    path_IN = os.path.join(path_script, PathIN,(Name+Filetype))
    img = cv.imread(path_IN)
    height, width, _ = img.shape
    
    # Smoothing / de-noising
    imgPMSF = cv.pyrMeanShiftFiltering(img, PyrFilt1, PyrFilt2)
    
    # Otsu binarization
    gray = cv.cvtColor(imgPMSF, cv.COLOR_BGR2GRAY)
    imgTSH = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    if ke != 0:  # erode-dilate
        kernel = np.ones((ke, ke), np.uint8)
        imgTSH = cv.dilate(cv.erode(imgTSH, kernel), kernel)
    
    # Contours
    cnts = cv.findContours(imgTSH.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for (i, c) in enumerate(cnts):
        ((x, y), _) = cv.minEnclosingCircle(c)
    
    # WATERSHED
    D = ndimage.distance_transform_edt(imgTSH)
    localMax = peak_local_max(D, indices=False, min_distance=MinDist, labels=imgTSH)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=imgTSH)
    
    # setup
    S = set([])  # shapes stats (radius)
    Shapes = []
    arr_out = np.zeros((height, width), dtype=int)
    img_out = np.zeros((height, width, 3), np.uint8)
    img_out[:] = Col_Background
    CX = 0
    
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
        CX+=1
    
        # SHAPE FITTING
        # draw enclosing circle 
        if not FitEllipse:
            ((x, y), r) = cv.minEnclosingCircle(c)
            Shapes.append([x, y, r])
            Shapes.append(r)
        # draw inscribed-bound ellipse
        if FitEllipse:
            # bounding box
            rect = (x, y), (w, h), Agl = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            # ellipse
            # (x,y),(a,b),Agl = cv.fitEllipse(box)
                # compute eccentricty
            if w > h:
                c = ((w ** 2) - (h ** 2)) ** (1 / 2)
                e = c / w
            else:
                c = ((h ** 2) - (w ** 2)) ** (1 / 2)
                e = c / h
            Shapes.append([x, y, w, h, Agl, c, e])
            S.add(c)
    
    # Discriminator
    S_med = stat.median(S)
    S_sigma = stat.stdev(S)
    S_avg = stat.mean(S)
    
    F = set([])  # same but filtered
    Fibers = []
    
    F_mean = S_med
    R_min = F_mean * (1 - F_RE)
    R_max = F_mean * (1 + F_RE)
    
    for shape in Shapes:
        # fit enclosing circle
        x, y = shape[0], shape[1]
        if not FitEllipse:
            r = shape[2]
            if R_min < r < R_max:
                F.add(r)
                cv.circle(arr_out, (int(x), int(y)), int(r), len(F), -1)
                # fiber ID and appending
                Fibers.append([len(F), round(x), round(y), round(r, 3)])
        if FitEllipse:
            a, b, Agl, c, e = shape[2:]
            if (a + b) > 2 * R_min and abs(e) < E_RE:
                F.add(c)
                cv.ellipse(arr_out, ((x, y), (a, b), Agl), len(F), -1)
                # fiber ID and appending
                Fibers.append([len(F), round(x), round(y), round(c, 3), round(e, 3)])

    # STATISCTICS:
    F_med = stat.median(F)
    F_sigma = stat.stdev(F)
    F_avg = stat.mean(F)
    
    if FitEllipse:
        E = []
        for ellipse in Fibers:
            E.append(ellipse[-1])
        E_avg = stat.mean(E)
        E_med = stat.median(E)
        E_sigma = stat.stdev(E)
     
    # OUTPUT:
    if Print_Output:
        print("\n\n -----")
        print('IMAGE TO FILE')
        path_script = os.path.dirname(__file__)
        path = os.path.join(path_script, path_R_output)
        os.chdir(path)
        print(os.listdir(path))
        print(path)
        filename = (output_file[0] + output_file[1])
        print(filename)
        cv.imwrite(filename, img_out)
        print('Successfully saved')
    
    if Print_Matrix:
        print("\n\n -----")
        print('MATRIX TO FILE')
        np.set_printoptions(threshold=np.inf)
        #print(arr_out)
        path_script = os.path.dirname(__file__)
        path = os.path.join(path_script, path_R_output)
        os.chdir(path)
        print(os.listdir(path))
        print(path)
        #numpy.savetxt((outpit_file[0]+output_file[2]),a,delimiter="")
        pd.DataFrame(arr_out).to_csv((path+output_file[2]), header="none",index="none")
        print('Successfully saved')
    
    print("\n\n -----")
    print('WATERSHED')
    print("[INFO] unique labels found :", format(CX))
    print("[INFO] unique shapes found   :", format(len(S)))
    print("[INFO] unique fibers found   :", format(len(F)))
    print("\n\n -----")
    print('SHAPES RADIUS')
    print("[INFO] median            :", round(S_med, 3))
    print("[INFO] mean              :", round(S_avg, 3))
    print("[INFO] standard deviation:", round(S_sigma, 3))
    print("\n\n -----")
    print('FIBERS RADIUS')
    print("[INFO] median            :", round(F_med, 3))
    print("[INFO] mean              :", round(F_avg, 3))
    print("[INFO] standard deviation:", round(F_sigma, 3))
    if FitEllipse:
        print("\n\n -----")
        print('FIBERS ECCENTRICITY')
        print("[INFO] median            :", round(E_med, 3))
        print("[INFO] mean              :", round(E_avg, 3))
        print("[INFO] standard deviation:", round(E_sigma, 3))
    return arr_out

# parameters:
input_file = ["TapeA", ".jpg"]
output_file = [input_file[0], ".png", "Watershed.csv"]
path_R_input = "../Data"
path_R_output = "../Data Processed/Watershed"
OUTPUT = WATERSHED(input_file[0], input_file[1], path_R_input, path_R_output) #(Name, Filetype, PathIN, pathOUT)
np.set_printoptions(threshold=np.inf)
#print(OUTPUT)
