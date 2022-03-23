# read image through command line
#from skimage.feature import peak_local_max
#from skimage.segmentation import watershed
#from scipy import ndimage
#import numpy as np
#import argparse
#import imutils
import cv2
img = cv2.imread(r"C:\Users\huege\Documents\GitHub\AE2223-I-DO8-Q3-4-\Data Processed\Tape A after WV2.jpg")

# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray_image)
# convert the grayscale image to binary image
ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
# find contours in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
for c in contours:
# calculate moments for each contour
    M = cv2.moments(c)
# calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print(cX,cY)
    cv2.circle(img, (cX, cY), 5, (20, 20, 20), -1)
    cv2.putText(img, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
# display the image

    cv2.imshow("Image", img)
    cv2.waitKey(0)
