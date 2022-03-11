import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

img = cv.imread(r'C:\Users\huege\Documents\GitHub\AE2223-I-DO8-Q3-4-\Data\TapeA_registration.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

cv.imshow('Otsu',ret)

# noise removal
kernel = np.ones((1,1),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations =2)
    # sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
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