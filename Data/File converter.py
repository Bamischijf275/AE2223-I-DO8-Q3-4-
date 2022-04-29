import geojson as GeoJSON
import numpy as np
from PIL import Image
import os
def convert_image():
#-------------------------------------------------------------------
# This function will convert the .jpg.tif files in Images and convert them to just .tif in image_tif
#-------------------------------------------------------------------

    print(os.listdir('Tape_B/Images'))
    directory = os.listdir("Tape_B/Images")
    print(directory)
    for i in directory:
        a = Image.open(f'Tape_B/Images/{i}')
        print(i)
        i_new = i.replace('.jpg','')
        a.save(f'Tape_B/Image_tif/{i_new}.tif')
    print("Done")

def masks_converter():
# -------------------------------------------------------------------
# This function will convert the .jpg.tif files in Masks and convert them to just .tif in masks_tif
# -------------------------------------------------------------------
    directory = os.listdir('Tape_B/Masks')
    for i in directory:
        a=Image.open(f'Tape_B/Masks/{i}')
        i_new = i.replace(".jpg","")
        print(i_new)
        a.save(f'Tape_B/Masks_tif/{i_new}')
    print("Done")


