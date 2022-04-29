import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt

directory = "GroundTruth"

def csv_to_numpy(filename):
#----------------------------------------------------
# Convert a csv file to a numpy array
#----------------------------------------------------
    file = open(filename)
    file_npar = np.loadtxt(file,delimiter=",")
    return file_npar


def combine_csv(j):
# ----------------------------------------------------
#Grab 10 patches of a picture, convert them to ndarray format
#Combine top and bottom with np.hstack, then add two horizntal patches wiht np.vstack
#Remove .jpg.tif from the filename as they are redundant
# ----------------------------------------------------
    new_ar_top = csv_to_numpy(f"GroundTruth/Tape_B_{j}_{1}.jpg.tif.csv")
    new_ar_bot = csv_to_numpy(f"GroundTruth/Tape_B_{j}_{6}.jpg.tif.csv")
    for i in range(2,6):
         print(np.shape(csv_to_numpy(f"GroundTruth/Tape_B_{j}_{i}.jpg.tif.csv")))
         new_ar_top = np.hstack((new_ar_top,csv_to_numpy(f"GroundTruth/Tape_B_{j}_{i}.jpg.tif.csv")))
    for h in range(7,11):
        new_ar_bot = np.hstack((new_ar_bot,csv_to_numpy(f"GroundTruth/Tape_B_{j}_{h}.jpg.tif.csv")))
    return_ar = np.vstack((new_ar_top,new_ar_bot))
    print(np.shape(return_ar))
    return return_ar

#Execute
#Comment = File has been executed, but several numbers are still missing here.
for o in range(1,10):
    a = combine_csv(o)

#   Visual check whether the pictures are "Correct"\

#   max = np.max(a)
#   plt.imshow(a,cmap = "gray")
#   plt.show()
    np.savetxt(f'GroundTruth/Merged CSV/Tape_B_{o}.csv',a,delimiter=",")