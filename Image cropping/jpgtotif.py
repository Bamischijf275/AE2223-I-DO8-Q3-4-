import numpy as np
from PIL import Image
import os

# for i in range(1,21):
#     im = Image.open(f"Uncropped pictures/Tape_B_2_{i}.jpg")
#     im_ar = np.array(im)
#     im_ar = im_ar[1:,1:]
#     im = Image.fromarray(im_ar)
#     im.save(f"Uncropped tif/Tape_B_2_{i}.tif")

for name in os.listdir("Data Processed/Watershed/Training"):
    if name != str("TIF"):
        name = name.replace('.csv', "")
        print(name)
        im = Image.open(f'Data/Tape_B/Tape_B_2_JPG/{name}.jpg')
        im_ar = np.array(im)
        print(np.shape(im_ar))
        im_ar_split = np.array_split(im_ar,5,axis=1)
        top = 1
        bot = 6
        for j in range(0,5):
            ar_topbot = im_ar_split[j]
            ar_topbot = np.array_split(ar_topbot,2,axis=0)
            ar_top = ar_topbot[0]
            ar_bot = ar_topbot[1]
            ar_top_im = Image.fromarray(ar_top)
            ar_bot_im = Image.fromarray(ar_bot)
            ar_top_im.save(f"Data Processed/Training/watershed/images/{name}_{top}.tif")
            ar_bot_im.save(f"Data Processed/Training/watershed/images/{name}_{bot}.tif")
            top+=1
            bot+=1