import cv2 as cv
import imutils
#import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import os
import statistics as stat
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import sys

# CONSTANTS
#parameters
input_file = "TapeB.png"
input_algo = ["Watershed","Annotated"]

# Display
Show_In = True
Show_Out = False

# IMAGE PROCESSING
# Open image
path_script = os.path.dirname(__file__)
path1 = os.path.join(path_script, input_algo[0], input_file)
img_1 = cv.imread(path1)
cv.imshow('INPUT ONE', img_1)

path2 = os.path.join(path_script, input_algo[1], input_file)
img_2 = cv.imread(path2)
cv.imshow('INPUT TWO', img_2)

print("\n\n -----")
print("path 1: ",path1)
print("path 2: ",path2)
im_h, im_w, _ = img_1.shape
print("image size: ", im_w, "x", im_h, "[pix]")

if Show_In: 
    cv.imshow('INPUT ONE', img_1)
    cv.imshow('INPUT TWO', img_2)

# COMPARISON

#find contours
gray = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cv.imshow("CHECK", thresh)
cnts = imutils.grab_contours(cnts)

for c in cnts:
	# compute the center of the contour
	M = cv.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the contour and center of the shape on the image
	cv.drawContours(img_1, [c], -1, (0, 255, 0), 2)
	cv.circle(img_1, (cX, cY), 7, (255, 255, 255), -1)
	# show the image
	cv.imshow("Image", img_1)
    
#find centers of mass
    
#circle fitting
    
#check for 1/2 radius
    
#classify

# STATISCTICS:

print("\n\n -----")
#np.set_printoptions(threshold=sys.maxsize)
print("\n\n -----")
print('SHAPES')
print("[INFO] median            :", round(1, 3))
print("[INFO] mean              :", round(2, 3))
print("[INFO] standard deviation:", round(3, 3))

#cv.imshow("Output", XXX)
cv.waitKey(0)