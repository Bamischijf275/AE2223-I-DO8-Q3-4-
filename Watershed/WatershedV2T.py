import cv2 as cv
import imutils
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import os
import statistics as stat
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed

# CONSTANTS:
# parameters:
F_mean = 5.4
F_RE = 0.3  # % deviation
F_perPixel = 0.7256  # 1)<- 2) 0.7232

R_min = F_mean * (1 - F_RE)
R_max = F_mean * (1 + F_RE)
MinDist = int(R_min - 1)

PyrFilt1 = 10  # iterations
PyrFilt2 = 10  # strength

# Display
Show_In = True
Show_Otsu = True
Show_PyrFilt = True
Show_Boundary = True
Show_ShapeCenter = True
Show_FiberCircle = True
Col_ShapeCenter = (255, 0, 0)
Col_Boundary = (0, 255, 0)
Col_FiberCircle = (0, 0, 255)

# IMAGE PROCESSING
# Open image
path_script = os.path.dirname(__file__)
path_relative = "..\Data\TapeB.tif"
path = os.path.join(path_script, path_relative)
img = cv.imread(path)
if Show_In: cv.imshow('INPUT', img)

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

S = set([])
F = set([])
Fibers = []
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
            cv.circle(img, (int(x), int(y)), int(r), Col_FiberCircle, 1)
            # cv.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
            # cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        F.add(r)
        Fibers.append([round(x), round(y), round(r, 1)])
        
# STATISCTICS:

S_med = stat.median(S)
S_sigma = stat.stdev(S)
S_avg = stat.mean(S)

F_med = stat.median(F)
F_sigma = stat.stdev(F)
F_avg = stat.mean(F)

print("\n\n -----")
for fib in Fibers:
    print("x, y, r: ", fib[0],fib[1], fib[2])
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
