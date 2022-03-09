import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from matplotlib import image as mpimg
img = cv.imread('Data/TapeA_registration.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('Data/TapeA_registration.jpg')

#imgplot=plt.imshow(img)
cv2.imshow('image window', image)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()