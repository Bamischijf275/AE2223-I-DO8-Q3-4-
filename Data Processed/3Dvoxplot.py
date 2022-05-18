
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
import open3d as o3d
import os
from skimage import io
from time import sleep
from tqdm import tqdm

def ar_compar(ar1,ar2):
    ar1_un = np.unique(ar1)
    ar2 = ar2*1000
    ar1_un = ar1_un[1:] #remove zero collum
    ar2_un = np.unique(ar2)
    ar2_un = ar2_un[1:]
    ar_re = np.zeros(np.shape(ar2))
    for value in ar1_un:
        a1 = np.argwhere(ar1 == value)
        amount = len(a1)
        score = 0
        number_store = np.array([])
        for indice in a1:
            if ar2[indice[0],indice[1]] !=0:
                score +=1
                number_store = np.append(number_store,int(ar2[indice[0],indice[1]]))
        if score/amount >=0.5:
            values, counts = np.unique(number_store,return_counts=True)
            ind = np.argmax(counts)
            a2 = np.argwhere(ar2 == values[ind])
            for pos in a2:
                ar_re[pos[0],pos[1]] = value
    return ar_re



def tif_generator():
    im_start = Image.open(f"Data Processed/AI 3D/Dataset3_v2/Tape_B_2_-{0}.jpg.tif")
    #ar = np.genfromtxt(f'Data Processed/Watershed/3D/Tape_B_2_-1L.csv',delimiter=",")
    ar = np.array(im_start)
    ar = ar[100:200,300:400]
    ar_last = ar
    for i in range(1,201):
        print(f"Tape_B_2_-{i}.jpg.tif")
        img = Image.open(f"Data Processed/AI 3D/Dataset3_v2/Tape_B_2_-{i}.jpg.tif")
        #img = np.genfromtxt(f'Data Processed/Watershed/3D/Tape_B_2_-{i}L.csv',delimiter=",")
        ar1 = np.array(img)
        ar1 = ar1[100:200,300:400]
        ar_add = ar_compar(ar_last,ar1)
        ar = np.dstack([ar,ar_add])
        ar_last = ar_add
    ar = np.swapaxes(ar,0,2)
    ar = np.swapaxes(ar,1,2)
    ar_un = np.unique(ar)
    ar_un = ar_un[1:]
    print(ar_un)
    for un in ar_un:
        ar_new = ar.copy()
        print(np.argwhere(ar_new==un),un)
        ar_new[ar_new!=un]=0
        ar_new[ar_new!=0] =1
        tifffile.imwrite(f'Data processed/3D/dataset3_V2/200-(0.5)/dataset3_V2_{un}.tif', ar_new.astype('uint16'), photometric="minisblack", imagej=True)
    print(np.shape(ar))
    tifffile.imwrite(f'Data processed/3D/dataset3_V2/200-(0.5)/dataset1_V3_whole.tif', ar.astype('uint16'),photometric= "minisblack",imagej=True)

tif_generator()
def extract_pos(filename): #test_27.0.tif
    im = io.imread(f'Data processed/3D tiff (0.5)/{filename}')
    ar = np.array(im)
    pos_ar = np.argwhere(ar !=0)
    pos_ar = np.array(pos_ar)
    if np.std(pos_ar[:,1]) >= 5 or np.std(pos_ar[:,2]) >=5:
        print(f"test_{i}.0.tif deviate")
