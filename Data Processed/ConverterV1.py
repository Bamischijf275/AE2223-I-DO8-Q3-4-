import cv2 as cv
import imutils
import math
import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import image as mpimg
import os
import pandas as pd
import statistics as stat
# from tqdm import tqdm
import sys
import time
import warnings
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)

File = [
    ["Tape_B", [2, 2], [17, 37], "name"],  # File
    ["../Data/Tape_B/Tape_B_2/",                ".tiff"], # IN
    ["../Data Processed/Watershed/Training/",   ".csv"]  #OUT
]

Program = ["full",  # fast-print-img-full-wait
           250,  # Substeps
           "ellipse"  # shape=circle,ellipse,contour
           ]

# LOOP
N = File[0][1]
M = File[0][2]

n = N[0]
while n <= N[1]:
    m = M[0]
    while m <= M[1]:
        print("\n ----- NEWFILE -----")
        name = File[0][0] + "_" + str(n) + "_-" + str(m)
        File[0][3] = name
        print("Image : ", name)

        Result = CONVERT(File, Program)
        m += 1
    n += 1
T11 = time.time()
print("\n ----- END PROGRAM ----- \n")
print("> " + str(round((T11 - T00), 1)) + "[s] <")

if Program[0] == "wait":
    cv.waitKey(0)
else: cv.waitKey(1)
