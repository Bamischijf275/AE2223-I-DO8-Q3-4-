import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os

for i in os.listdir("manual/images"):
    im = Image.open(f"manual/images/{i}",'r')
    mask = Image.open(f"manual/masks/{i}",'r')
    if im.size != mask.size:
        print(im.size,mask.size,i)
        im_ar = np.array(im)
        im_ar = im_ar[:,2:]
        im_new = Image.fromarray(im_ar)
        print(im_new)
        print(im_new.size,mask.size,i)
        im_new.save(f'{i}')


