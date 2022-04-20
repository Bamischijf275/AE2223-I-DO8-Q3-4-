import geojson as GeoJSON
import numpy as np
from PIL import Image
import os

directory = os.listdir("Tape_B/Masks")
print(directory)
for i in directory:
    i = str(i)
    ie = str(os.path.join("Tape_B/Masks",i))
    img = Image.open(ie)
    img_ar = np.asarray(img)
    np.savetxt(f'Tape_B/CSV_Masks/{i}.csv',img_ar,delimiter=" , ")


