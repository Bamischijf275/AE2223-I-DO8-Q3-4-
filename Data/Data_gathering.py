import geojson as GeoJSON
import numpy as np
from PIL import Image
import os
# print(os.listdir('Tape_B/Images'))
# directory = os.listdir("Tape_B/Images")
# print(directory)
# for i in directory:
#     a = Image.open(f'Tape_B/Images/{i}')
#     print(i)
#     i_new = i.replace('.jpg','')
#     a.save(f'Tape_B/Image_tif/{i_new}.tif')

directory = os.listdir('Tape_B/Images')
a = Image.open("Tape_B/Images/Tape_B_11_1.jpg.tif")
a.save("Tape_B/Image_tif/Tape_B_11_1.tif")
# for i in directory:
#         print(i)
#         a=Image.open(f'Tape_B/Images/{i}')
#         i_new = i.replace(".jpg","")
#         a.save(f'Tape_B/Image_tif/{i_new}.tif')

