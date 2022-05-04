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

T00 = time.time()
    

# MAIN LOOP
print("----- *-* -----")
print("----- START PROGRAM -----")
print("----- *-* -----")

T00 = time.time()

#Algos
Algorithms = ["Test","Watershed"]

#files
Name = "Tape_B"

N=[1,1]
M=[4,4]

Info = ["Training/comparing/masks",
        "Annotated"]
Type=[".tif",".csv"]

n = N[0]
while n <= N[1]:
    m=M[0]
    while m <= M[1]:
        print("\n NEWFILE")
        name = Name+"_"+str(n)+"_"+str(m)
        print(str(name))
        #extract matrices
        pathScript=os.path.dirname(__file__)
        
        path = Info[0] + "/" + name + Type[0]
        path =  os.path.join(pathScript,path)
        print(path)
        
        Img = cv.imread(path)
        cv.imshow('Image', Img)
        cv.waitKey(1)
        
        Arr = np.array(Img)
        
        path = Info[1] + "/" + name + Type[1]
        path =  os.path.join(pathScript,path)
        print(path)
        
        pd.DataFrame(Arr).to_csv((path), header="none", index="none")
        
    print("\n ENDFILE \n")
    m+=1
n+=1
        
T11 = time.time()
print("> " + str(round((T11 - T00),1)) + "[s] <")
print("\n----- END PROGRAM ----- \n")