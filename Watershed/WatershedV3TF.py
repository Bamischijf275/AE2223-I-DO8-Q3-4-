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
#import sys

def WATERSHED (Name, Filetype, PathIN, pathOUT):

    # PARAMETER DEFINITION
   # Watershed
    F_mean = 5
    F_RE = 1 / 3  # strandard deviations
    E_RE = 0.85
    R_min = F_mean * (1 - F_RE)
    MinDist = int(R_min - 1)
    
    PyrFilt1 = 14  # iterations
    PyrFilt2 = 12  # strength
    
    # Extra Processing
    ke = 3  # kernel
    FitEllipse = True
    
    #Display
    Col_Background = (100, 100, 100)
    
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
    
    # FIBERS
    # setup
    S = set([])  # shapes stats (radius)
    Shapes = []
    arr_out = np.zeros((height, width), dtype=int)
    img_out = np.zeros((height, width, 3), np.uint8)
    img_out[:] = Col_Background
    
    for label in np.unique(labels):
        # eliminate background
        if label == 0:
            continue
    
        # mask and label region
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
    
        # find contours
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
    
        # SHAPE FITTING
        # draw enclosing circle 
        if not FitEllipse:
            circle = ((x, y), r) = cv.minEnclosingCircle(c)
            Shapes.append([x, y, r])
            Shapes.append(r)
        # draw inscribed-bound ellipse
        if FitEllipse:
            # bounding box
            rect = (x, y), (w, h), Agl = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            # ellipse
            # compute eccentricty
            if w > h:
                c = ((w ** 2) - (h ** 2)) ** (1 / 2)
                e = c / w
            else:
                c = ((h ** 2) - (w ** 2)) ** (1 / 2)
                e = c / h
            Shapes.append([x, y, w, h, Agl, c, e])
            S.add(c)
    
        # fiber ID and appending
        S.add(r)
        if R_min < r < R_max:
            cv.circle(arr_out, (int(x), int(y)), int(r),  len(F), -1)
            F.add(r)
            Fibers.append([len(F),round(x), round(y), round(r, 1)])

    # OUTPUT:
    return arr_out

# parameters:
input_file = ["TapeA",".jpg"]
output_file = [input_file[0], ".png"]
path_R_input = "../Data"
path_R_output = "../Data Processed/Watershed"
OUTPUT = WATERSHED(input_file[0], input_file[1], path_R_input, path_R_output) #(Name, Filetype, PathIN, pathOUT)
np.set_printoptions(threshold=np.inf)
print(OUTPUT)
