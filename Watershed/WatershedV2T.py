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

# CONSTANTS:
# parameters:
F_mean = 5.4
F_RE = 0.3  # % deviation

R_min = F_mean * (1 - F_RE)
R_max = F_mean * (1 + F_RE)
MinDist = int(R_min - 1)

PyrFilt1 = 10  # iterations
PyrFilt2 = 10  # strength

input_file = ["TapeB",".tif"]
output_file = [input_file[0], ".png"]
path_R_input = "../Data"
path_R_output = "../Data Processed/Watershed"

# Display
Show_In = False
Show_Otsu = False
Show_PyrFilt = False
Show_Boundary = False
Show_ShapeCenter = False
Show_FiberCircle = False
Col_ShapeCenter = (255, 0, 0)
Col_Boundary = (0, 255, 0)
Col_FiberCircle = (0, 0, 255)

Print_Output = True

# IMAGE PROCESSING
# Open image
path_script = os.path.dirname(__file__)
path = os.path.join(path_script, path_R_input,(input_file[0]+input_file[1]))
#print (path)
img = cv.imread(path)
if Show_In: cv.imshow('input', img)
height, width, _ = img.shape

# Smoothing / de-noising
imgPMSF = cv.pyrMeanShiftFiltering(img, PyrFilt1, PyrFilt2)
if Show_PyrFilt: cv.imshow('imagePMSF', imgPMSF)

# Otsu binarization
gray = cv.cvtColor(imgPMSF, cv.COLOR_BGR2GRAY)
imgTSH = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
if Show_Otsu:
    cv.imshow("Thresh", imgTSH)

# Contours
cnts = cv.findContours(imgTSH.copy(), cv.RETR_EXTERNAL,
                       cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for (i, c) in enumerate(cnts):
    ((x, y), _) = cv.minEnclosingCircle(c)
    if Show_Boundary:
        cv.drawContours(img, [c], -1, Col_Boundary, 1)
# cv.putText(img, "#{}".format(i + 1), (int(x) - 10, int(y)),
# cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# WATERSHED
D = ndimage.distance_transform_edt(imgTSH)
localMax = peak_local_max(D, indices=False, min_distance=MinDist, labels=imgTSH)
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=imgTSH)

# FIBERS

S = set([]) #shapes stats (radius)
F = set([]) #same but filtered
Fibers = []
img_out = np.zeros((height,width,3), np.uint8)
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
    if Show_ShapeCenter:
        cv.circle(img, (int(x), int(y)), 1, Col_ShapeCenter, 0)

    # fiber ID and appending
    S.add(r)
    if R_min < r < R_max:
        if Show_FiberCircle:
            cv.circle(img, (int(x), int(y)), int(r), (int((x/width)*255),int((y/height)*255),int(500*(r-F_mean)**2)),-1)
        cv.circle(img_out, (int(x), int(y)), int(r), (int((x/width)*255),int((y/height)*255),int(500*(r-F_mean)**2)),-1)
        F.add(r)
        Fibers.append([len(Fibers),round(x), round(y), round(r, 1)])
        
# OUTPUT:
#array_out = np.zeros((height,width), np.uint16)
#for fib in Fibers:
    #print("x, y, r: ", fib[0],fib[1], fib[2])
    
if Print_Output:
    print("\n\n -----")
    print('IMAGE TO FILE')
    path_script = os.path.dirname(__file__)
    path = os.path.join(path_script, path_R_output)
    os.chdir(path)
    print(os.listdir(path))
    print(path)
    filename = (output_file[0]+output_file[1])
    print(filename)
    cv.imwrite(filename, img_out)
    print('Successfully saved')
    
# STATISCTICS:

S_med = stat.median(S)
S_sigma = stat.stdev(S)
S_avg = stat.mean(S)

F_med = stat.median(F)
F_sigma = stat.stdev(F)
F_avg = stat.mean(F)

print("\n\n -----")
print('WATERSHED')
print("[INFO] {} unique contours found".format(len(cnts)))
print("[INFO] {} unique segments found".format(len(S)))
print("[INFO] {} unique fibers found".format(len(F)))
print("\n\n -----")
print('SHAPES')
print("[INFO] median            :", round(S_med, 3))
print("[INFO] mean              :", round(S_avg, 3))
print("[INFO] standard deviation:", round(S_sigma, 3))
print("\n\n -----")
print('FIBERS')
print("[INFO] median            :", round(F_med, 3))
print("[INFO] mean              :", round(F_avg, 3))
print("[INFO] standard deviation:", round(F_sigma, 3))

cv.imshow("Output", img)
cv.waitKey(0)
