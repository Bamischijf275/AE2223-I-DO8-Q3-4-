import geojson as GeoJSON
import numpy as np
from PIL import Image
import os

# directory = os.listdir("Tape_B/Masks")
# print(directory)
# for i in directory:
#     i = str(i)
#     ie = str(os.path.join("Tape_B/Masks",i))
#     i.replace(".jpg.tif","")
#     img = Image.open(ie)
#     img_ar = np.asarray(img)
#     np.savetxt(f'Tape_B/CSV_Masks/{i}.csv',img_ar,delimiter=" , ")

for i in os.listdir("Tape_B/CSV_Masks"):
    i_new = i.replace(".jpg.tif","")
    os.rename(f'Tape_B/CSV_Masks/{i}',f'Tape_B/CSV_Masks/{i_new}')

