# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
#import argparse
import imutils
import cv2
import os
from PIL import Image as uwu

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())
# load the image and perform pyramid mean shift filtering
# to aid the thresholding step

cwd = os.getcwd()
files = os.listdir(cwd)
print(cwd,files)

image = cv2.imread(r"Image cropping\Cropped data\Tape_B\Tape_B_4\Tape_B_4_2.jpg")
shifted = cv2.pyrMeanShiftFiltering(image, 1, 2)
cv2.imshow("Input", image)
# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=2,
	labels=thresh)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
# loop over the unique mask returned by the Watershed
# algorithm
height,width, _ = image.shape
#height=232
#width=952
blank_image = np.zeros((height,width,3), np.uint8)

for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(blank_image, (int(x), int(y)), int(r), (abs(2*int(x)-0.5*int(y))*12, (int(x)+int(y))/4, 3*int(x)*int(y)/(int(x)+int(y))), -1)

	#cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# show the output image
cv2.imshow("Output", image)
#print(blank_image)

cv2.imshow("no bg", blank_image)


cv2.imwrite('C:Data\Tabe_B_1_1.jpg', blank_image)

cv2.waitKey(0)
