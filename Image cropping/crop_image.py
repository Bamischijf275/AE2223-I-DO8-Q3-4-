import numpy
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import os
import random

cwd = os.getcwd()
files = os.listdir(cwd)
print(cwd,files)
column_crop = 5
width_crop = 2

def image_cropper(img,filename,num):
    im = Image.open(img,mode='r').convert("L")
    # try:
    #     os.makedirs(f'Image cropping/Cropped data/{tapename}/{direc}')
    # except OSError:
    #     print("ERROR")
    # else:
    #     print("Made new folder")
    im_array = numpy.asarray(im)
    height = round(np.shape(im_array)[1]/column_crop)
    width = round(np.shape(im_array)[0]/width_crop)
    print(height,width)
    im1_ar = im_array[:width,:height]
    im2_ar = im_array[:width,height:2*height]
    im3_ar = im_array[:width,2*height:3*height]
    im4_ar = im_array[:width,3*height:4*height]
    im5_ar = im_array[:width,4*height:]
    im6_ar = im_array[width:,:height]
    im7_ar = im_array[width:,height:2*height]
    im8_ar = im_array[width:,2*height:3*height]
    im9_ar = im_array[width:,3*height:4*height]
    im10_ar = im_array[width:,4*height:]

    im1 = Image.fromarray(im1_ar)
    im2 = Image.fromarray(im2_ar)
    im3 = Image.fromarray(im3_ar)
    im4 = Image.fromarray(im4_ar)
    im5 = Image.fromarray(im5_ar)
    im6 = Image.fromarray(im6_ar)
    im7 = Image.fromarray(im7_ar)
    im8 = Image.fromarray(im8_ar)
    im9 = Image.fromarray(im9_ar)
    im10 = Image.fromarray(im10_ar)
    filename = filename.replace(".jpg","")
    filename = filename.replace("2_","")
    print(filename)

    im1.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_1.tif')
    im2.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_2.tif')
    im3.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_3.tif')
    im4.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_4.tif')
    im5.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_5.tif')
    im6.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_6.tif')
    im7.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_7.tif')
    im8.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_8.tif')
    im9.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_9.tif')
    im10.save(f'Data Processed/Watershed/Training/{num} (tif)/images/{filename}_10.tif')



def image_opener(direc_path):
    for i in range(1,21):
        j = random.randint(1,2189)
        try:
            img = Image.open(f'{direc_path}/Tape_B_2_-{j}.jpg')
            img.save(f'Image cropping/Uncropped pictures/Tape_B_2_{i}.jpg')
        except EOFError:
            print( "ERROR")
            return False



#image_opener(r'C:\Users\degro\OneDrive - Delft University of Technology\Aerospace engineering\Year 2\Q3-4 project\Data\Data\Training data\Tape_B_2_JPG')

for name in os.listdir(f'Data Processed/Watershed/Training/100/Images'):
     image_cropper(f'Data Processed/Watershed/Training/100/images/{name}',name,100)