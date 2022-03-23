import geojson as GeoJSON
import numpy as np
from PIL import Image

img = Image.open("Data\Tape_B_1_1.jpg.tif")
print(img)
img_ar = np.asarray(img)
print(img_ar)
print(np.max(img_ar))