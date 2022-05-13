
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
import open3d as o3d
import os
from skimage import io
from time import sleep
from tqdm import tqdm

img1 = Image.open(f"Data Processed/AI 3D/Tape_B_2_-{0}.jpg.tif")
ar1 = np.array(img1)
img2 = Image.open(f"Data Processed/AI 3D/Tape_B_2_-{1}.jpg.tif")
ar2 = np.array(img2)
ar2 = ar2
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
        if score/amount >=0.7:
            values, counts = np.unique(number_store,return_counts=True)
            ind = np.argmax(counts)
            a2 = np.argwhere(ar2 == values[ind])
            for pos in a2:
                ar_re[pos[0],pos[1]] = value
    return ar_re



def tif_generator():
    im_start = Image.open(f"Data Processed/AI 3D/Tape_B_2_-{0}.jpg.tif")
    ar = np.array(im_start)
    ar = ar
    ar_last = ar
    for i in range(1,500):
        print(f"Tape_B_2_-{i}.jpg.tif")
        img = Image.open(f"Data Processed/AI 3D/Tape_B_2_-{i}.jpg.tif")
        ar1 = np.array(img)
        ar1 = ar1
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
        tifffile.imwrite(f'Data processed/3D tiff/test_{un}.tif', ar_new.astype('uint16'), photometric="minisblack", imagej=True)
    print(np.shape(ar))
    tifffile.imwrite(f'Data processed/3D tiff/test.tif', ar.astype('uint16'),photometric= "minisblack",imagej=True)


def extract_pos(filename): #test_27.0.tif
    im = io.imread(f'Data processed/3D tiff/{filename}')
    ar = np.array(im)
    pos_ar = np.argwhere(ar !=0)
    pos_ar = np.array(pos_ar)
    if np.std(pos_ar[:,1]) >= 5 or np.std(pos_ar[:,2]) >=5:
        print(f"test_{i}.0.tif deviate")
for i in tqdm(range(1,946)):
    extract_pos(str(f"test_{i}.0.tif"))