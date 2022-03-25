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

# import sys

# PARAMETERS:
# System:
input_file = ["TapeA", ".jpg"]
output_file = [input_file[0], ".png", ".csv"]
path_R_input = "../Data"
path_R_output = "../Data Processed/Watershed"

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

# Display
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

Print_Matrix = True
Print_Output = False

# IMAGE PROCESSING
# Open image
path_script = os.path.dirname(__file__)
path = os.path.join(path_script, path_R_input, (input_file[0] + input_file[1]))
img = cv.imread(path)
if Show_In: cv.imshow('Input', img)
height, width, _ = img.shape

# Smoothing / de-noising
imgPMSF = cv.pyrMeanShiftFiltering(img, PyrFilt1, PyrFilt2)
if Show_PyrFilt: cv.imshow('imagePMSF', imgPMSF)
# Otsu binarization
gray = cv.cvtColor(imgPMSF, cv.COLOR_BGR2GRAY)
imgTSH = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
if ke != 0:  # erode-dilate
    kernel = np.ones((ke, ke), np.uint8)
    imgTSH = cv.dilate(cv.erode(imgTSH, kernel), kernel)
if Show_Otsu: cv.imshow("Thresh", imgTSH)

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
img_B = cv.imread(path)
img_S = cv.imread(path)
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
    if Show_Boundary:
        cntsX = cv.findContours(imgTSH.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cntsX = imutils.grab_contours(cntsX)
        for (i, cX) in enumerate(cnts):
            CX += 1
            cv.drawContours(img_B, [cX], -1, Col_Boundary, 1)

    # SHAPE FITTING
    # draw enclosing circle 
    if not FitEllipse:
        circle = ((x, y), r) = cv.minEnclosingCircle(c)
        Shapes.append([x, y, r])
        if Show_Shapes and Show_Fitted and not FitEllipse:
            Col = (int((x / width) * 255), int((y / height) * 255), int(500 * (r - F_mean) ** 2))
            cv.circle(img_S, (int(x), int(y)), int(r), Col, 1)
        Shapes.append(r)
    # draw inscribed-bound ellipse
    if FitEllipse:
        # bounding box
        rect = (x, y), (w, h), Agl = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        # ellipse
        # (x,y),(a,b),Agl = cv.fitEllipse(box)
        if Show_Shapes and Show_Fitted:
            Col = (int((x / width) * 255), int((y / height) * 255), int(200 * (w - h) ** 2))
            img_S = cv.ellipse(img_S, ((x, y), (w, h), Agl), Col, 1)
            img_S = cv.drawContours(img_S, [box], 0, Col, 1)
            # compute eccentricty
            if w > h:
                c = ((w ** 2) - (h ** 2)) ** (1 / 2)
                e = c / w
            else:
                c = ((h ** 2) - (w ** 2)) ** (1 / 2)
                e = c / h
        Shapes.append([x, y, w, h, Agl, c, e])
        S.add(c)

    # draw center
    if Show_Center:
        cv.circle(img_S, (int(x), int(y)), 1, Col_ShapeCenter, -1)

if Show_Boundary: cv.imshow("Boundaries", img_B)

# Discriminator
S_med = stat.median(S)
S_sigma = stat.stdev(S)
S_avg = stat.mean(S)

F = set([])  # same but filtered
Fibers = []
img_F = cv.imread(path)

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
            if Show_Fitted:
                Col = (int((x / width) * 255), int((y / height) * 255), int(500 * (r - F_mean) ** 2))
                cv.circle(img_F, (int(x), int(y)), int(r), Col, -1)
                cv.circle(img_out, (int(x), int(y)), int(r), Col, -1)
            if Show_Center:
                cv.circle(img_F, (int(x), int(y)), 1, Col_ShapeCenter, -1)
            # fiber ID and appending
            Fibers.append([len(F), round(x), round(y), round(r, 3)])
    if FitEllipse:
        a, b, Agl, c, e = shape[2:]
        if (a + b) > 2 * R_min and abs(e) < E_RE:
            F.add(c)
            cv.ellipse(arr_out, ((x, y), (a, b), Agl), len(F), -1)
            if Show_Fitted:
                Col = (int((x / width) * 255), int((y / height) * 255), int(500 * (c) ** 2))
                cv.ellipse(img_F, ((x, y), (a, b), Agl), Col, -1)
                cv.ellipse(img_out, ((x, y), (a, b), Agl), Col, -1)
            if Show_Center:
                cv.circle(img_F, (int(x), int(y)), 1, Col_ShapeCenter, -1)
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
    np.set_printoptions(threshold=np.inf)
    #print(arr_out)
    path_script = os.path.dirname(__file__)
    path = os.path.join(path_script, path_R_output, ("Waterhsed" + output_file[2]))
    path=(r"C:/Users/huege/Documents/GitHub/AE2223-I-DO8-Q3-4-/Data Processed/Watershed/Watershed.csv")
    print("\n\n -----")
    print(path)
    #os.chdir(path)
    print("\n\n -----")
    #numpy.savetxt((outpit_file[0]+output_file[2]),a,delimiter="")
    pd.DataFrame(arr_out).to_csv(path, header="none",index="none")

print("\n\n -----")
print('WATERSHED')
print("[INFO] unique contours found :", format(CX))
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

if Show_Shapes: cv.imshow("Shapes", img_S)
if Show_Fibers: cv.imshow("Fibers", img_F)
cv.imshow("OUTPUT", img_out)
cv.waitKey(0)
