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

def image_cropper(img,direc):
    im = Image.open(img,mode='r')
    try:
        os.makedirs(f'Image cropping/Cropped data/Tape_B/{direc}')
    except OSError:
        print("ERROR")
    else:
        print("Made new folder")
    im_array = numpy.asarray(im)
    print(np.shape(im_array))
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

    im1.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_1.jpg')
    im2.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_2.jpg')
    im3.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_3.jpg')
    im4.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_4.jpg')
    im5.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_5.jpg')
    im6.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_6.jpg')
    im7.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_7.jpg')
    im8.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_8.jpg')
    im9.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_9.jpg')
    im10.save(f'Image cropping/Cropped data/Tape_B/{direc}/{direc}_10.jpg')



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

for i in range(1,21):
     image_cropper(f'Image cropping/Uncropped pictures/Tape_B_2_{i}.jpg',f'Tape_B_{i}')