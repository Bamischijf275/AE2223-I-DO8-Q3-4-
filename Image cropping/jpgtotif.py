import numpy as np
from PIL import Image


for i in range(1,21):
    im = Image.open(f"Uncropped pictures/Tape_B_2_{i}.jpg")
    im_ar = np.array(im)
    im_ar = im_ar[1:,1:]
    im = Image.fromarray(im_ar)
    im.save(f"Uncropped tif/Tape_B_2_{i}.tif")