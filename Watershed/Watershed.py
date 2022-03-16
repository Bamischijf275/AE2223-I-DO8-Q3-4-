import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os

#PARAMETERS:
k_Kernel = 1
k_OPN = 100
k_DIL = k_OPN + 1
K_dist = 0.5

#open image
path_script = os.path.dirname(__file__)
path_relative = "..\Data\TapeB.tif"
path = os.path.join(path_script, path_relative)
img = cv.imread(path)

#Otsu binarization
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#BG/FG/X separation
kernel = np.ones((k_Kernel,k_Kernel),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations =k_OPN)

sure_bg = cv.dilate(opening,kernel,iterations=k_DIL)

dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,K_dist*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

#cv.imshow('sureBG',sure_bg)
#cv.imshow('sureFG',sure_fg)
#cv.imshow('Unknown',unknown)

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0

#Watershed and output
markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]
print(len(markers)-1)
cv.imshow('image',img)
#END
cv.waitKey(0)