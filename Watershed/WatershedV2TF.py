import cv2 as cv
import imutils
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import os
#import statistics as stat
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
#import sys

def WATERSHED (Name, Filetype, PathIN, pathOUT):

    # PARAMETER DEFINITION
    F_mean = 5.4
    F_RE = 0.3  # % deviation
    
    PyrFilt1 = 10  # iterations
    PyrFilt2 = 10  # strength
    
    R_min = F_mean * (1 - F_RE)
    R_max = F_mean * (1 + F_RE)
    MinDist = int(R_min - 1)
    
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
    #definitions
    S = set([]) #shapes stats (radius)
    F = set([]) #same but filtered
    Fibers = []
    arr_out = np.zeros((height,width), dtype=int)
    
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
    
        # fit enclosing circle
        ((x, y), r) = cv.minEnclosingCircle(c)
    
        # fiber ID and appending
        S.add(r)
        if R_min < r < R_max:
            cv.circle(arr_out, (int(x), int(y)), int(r),  len(F), -1)
            F.add(r)
            Fibers.append([len(F),round(x), round(y), round(r, 1)])
            
    # OUTPUT:
    return arr_out

# parameters:
input_file = ["TapeB",".tif"]
output_file = [input_file[0], ".png"]
path_R_input = "../Data"
path_R_output = "../Data Processed/Watershed"
OUTPUT = WATERSHED(input_file[0], input_file[1], path_R_input, path_R_output) #(Name, Filetype, PathIN, pathOUT)
print(OUTPUT)
