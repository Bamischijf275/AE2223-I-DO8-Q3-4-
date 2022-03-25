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

# PARAMETERS:
# System:
input_file = ["TapeA",".jpg"]
output_file = [input_file[0], ".png"]
path_R_input = "../Data"
path_R_output = "../Data Processed/Watershed"

# Watershed
F_mean = 5.2
F_RE = 0.3  # % deviation
R_min = F_mean * (1 - F_RE)
R_max = F_mean * (1 + F_RE)
MinDist = int(R_min-1)

PyrFilt1 = 10  # iterations
PyrFilt2 = 10  # strength

# Display
Show_In = True
Show_Otsu = True
Show_PyrFilt = True
Show_Shapes = True
Show_Fibers = True

Show_Boundary = True
Show_Circle = True
Show_Center = True
Col_ShapeCenter = (255, 0, 0)
Col_Boundary = (0, 255, 0)
Col_FiberCircle = (0, 0, 255)
Col_Background = (100,100,100)

Print_Matrix = False
Print_Output = True

# IMAGE PROCESSING
# Open image
path_script = os.path.dirname(__file__)
path = os.path.join(path_script, path_R_input,(input_file[0]+input_file[1]))
#print (path)
img  = cv.imread(path)
if Show_In: cv.imshow('Input', img)
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
cnts = cv.findContours(imgTSH.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
C = 0
# Shape fitting
for (i, c) in enumerate(cnts):
    #circle
    ((x, y), _) = cv.minEnclosingCircle(c)
    #end
    C+=1
    if Show_Boundary: cv.drawContours(img, [c], -1, Col_Boundary, 1)

# WATERSHED
D = ndimage.distance_transform_edt(imgTSH)
localMax = peak_local_max(D, indices=False, min_distance=MinDist, labels=imgTSH)
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=imgTSH)

# setup
S = set([]) #shapes stats (radius)
F = set([]) #same but filtered
Fibers = []
Shapes = []
arr_out = np.zeros((height,width), dtype=int)
img_out = np.zeros((height,width,3), np.uint8)
img_out[:]=Col_Background
if Show_Boundary : cv.imshow("Boundary", img)
img_S = cv.imread(path)

# for each identified center area
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

    # draw enclosing circle
    ((x, y), r) = cv.minEnclosingCircle(c)
    if Show_Shapes:
        if Show_Circle:
            cv.circle(img_S, (int(x), int(y)), int(r), (int((x/width)*255),int((y/height)*255),int(500*(r-F_mean)**2)),-1)
        if Show_Center:
            cv.circle(img_S, (int(x), int(y)), 1, Col_ShapeCenter, -1)
    # shape ID and appending
    S.add(r)
    Shapes.append([x, y, r])

#Discriminator
S_med = stat.median(S)
S_sigma = stat.stdev(S)
S_avg = stat.mean(S)

F_mean = S_avg

R_min = F_mean * (1 - F_RE)
R_max = F_mean * (1 + F_RE)
MinDist = int(R_min)

#Second run (fibers only)
img_F  = cv.imread(path)
for shape in Shapes:
    # fit enclosing circle
    x,y,r = shape
    if R_min < r < R_max:
        cv.circle(arr_out, (int(x), int(y)), int(r),  len(F), -1)
        if Show_Circle:
            cv.circle(img_F, (int(x), int(y)), int(r), (int((x/width)*255),int((y/height)*255),int(500*(r-F_mean)**2)),-1)
            cv.circle(img_out, (int(x), int(y)), int(r), (int((x/width)*255),int((y/height)*255),int(500*(r-F_mean)**2)),-1)
        if Show_Center:
            cv.circle(img_F, (int(x), int(y)), 1, Col_ShapeCenter, -1)
        # fiber ID and appending
        Fibers.append([len(F),round(x), round(y), round(r, 1)])
        F.add(r)

# OUTPUT:

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

F_med = stat.median(F)
F_sigma = stat.stdev(F)
F_avg = stat.mean(F)

if Print_Matrix:
    np.set_printoptions(threshold=np.inf)
    print(arr_out)
    
print("\n\n -----")
print('WATERSHED')
print("[INFO] {} unique contours found".format(C))
print("[INFO] {} unique shapes found".format(len(S)))
print("[INFO] {} unique fibers found".format(len(F)))
print("\n\n -----")
print('SHAPES')
print("[INFO] median            :", round(S_med, 3))
print("[INFO] mean              :", round(S_avg, 3))
print("[INFO] standard deviation:", round(S_sigma, 3))
print("\n\n -----")
print('FIBERS DISCRIMINATION')
print("[INFO] median            :", round(F_med, 3))
print("[INFO] mean              :", round(F_avg, 3))
print("[INFO] standard deviation:", round(F_sigma, 3))

if Show_Shapes: cv.imshow("Shapes", img_S)
if Show_Fibers: cv.imshow("Fibers", img_F)
cv.imshow("OUTPUT", img_out)
cv.waitKey(0)
