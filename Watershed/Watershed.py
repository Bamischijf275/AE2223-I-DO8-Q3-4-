import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

img = cv.imread(r'C:\Users\huege\Documents\GitHub\AE2223-I-DO8-Q3-4-\Data\TapeA_registration.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 5)
    # sure background area
sure_fg = cv.dilate(opening,kernel,iterations=5)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_bg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_bg = np.uint8(sure_bg)
unknown = cv.subtract(sure_bg,sure_fg)

cv.imshow('sureBG',sure_bg)
cv.imshow('sureFG',sure_fg)
cv.imshow('Unknown',unknown)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv.imshow('image',img)
cv.waitKey(0)