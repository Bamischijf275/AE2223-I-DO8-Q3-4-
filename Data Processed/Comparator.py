import cv2 as cv
#import imutils
#import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import os
#import statistics as stat
#from scipy import ndimage
#from skimage.feature import peak_local_max
#from skimage.segmentation import watershed
#import sys
import cv2


# CONSTANTS
#parameters
input_file = "TapeB"
input_algo = ["Watershed","Ground Thruth"]

# Display
Show_In = False
Show_Out = False

# IMAGE PROCESSING
# Open image
path_script = os.path.dirname(__file__)

for image in input_compare:
    path_R_input = "..\Data Processed\TapeB_WT-V2T.png"
path = os.path.join(path_script, path_R_input_GT)
img_GT = cv.imread(path)
path = os.path.join(path_script, path_R_input_WT)
img_WT = cv.imread(path)
#path = os.path.join(path_script, path_R_input_AI)
#img_AI = cv.imread(path)
if Show_In: 
    cv.imshow('INPUT Ground Truth', img_GT)
    height_GT, width_GT, _ = img_GT.shape

# COMPARISON

# STATISCTICS:

print("\n\n -----")
#np.set_printoptions(threshold=sys.maxsize)
print("\n\n -----")
print('SHAPES')
print("[INFO] median            :", round(1, 3))
print("[INFO] mean              :", round(2, 3))
print("[INFO] standard deviation:", round(3, 3))

cv.imshow("Output", img)
cv.waitKey(0)