import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

def ID_renamer(ar):
    ID = 0
    for j in np.unique(ar):
        if j !=0:
            ID +=1
            ar[ar==j] = ID
    return ar
#
# for name in os.listdir("Watershed/Training"):
#     print(f"{name}")
#     if name != str("TIF"):
#         ar = np.genfromtxt(f"Watershed/Training/{name}",delimiter=",")
#         filename = name.replace(".csv","")
#         ar = ar[1:,1:]
#         ar = ar
#         ar_split = np.array_split(ar,5,axis=1)
#         top = 1
#         bot = 6
#         for j in range(0,5):
#             ar_topbot = ar_split[j]
#             ar_topbot = np.array_split(ar_topbot,2,axis=0)
#             ar_top = ar_topbot[0].astype("uint16")
#             ar_bot = ar_topbot[1].astype("uint16")
#             ar_top = ID_renamer(ar_top)
#             ar_bot = ID_renamer(ar_bot)
#             ar_top_im = Image.fromarray(ar_top,)
#             ar_bot_im = Image.fromarray(ar_bot)
#             ar_top_im.save(f'Watershed/Training/TIF/{filename}_{top}.tif')
#             ar_bot_im.save(f'Watershed/Training/TIF/{filename}_{bot}.tif')
#             top +=1
#             bot +=1





# ID=0
# for name in os.listdir("Data Processed/Training/watershed/images"):
#     a_m = Image.open(f'Data Processed/Training/watershed/masks/{name}')
#     a_im = Image.open(f'Data Processed/Training/watershed/images/{name}')
#     a_m_ar = np.array(a_m)
#     a_im_ar = np.array(a_im)
#     if np.shape(a_m_ar) != np.shape(a_im_ar):
#         print(np.shape(a_m_ar),np.shape(a_im_ar))
#         print(f"Placeholder {ID},{name}")
#     # fig = plt.figure(figsize=(5,6))
#     # fig.add_subplot(2,1,1)
#     # plt.imshow(a_m_ar,cmap = "gray")
#     # fig.add_subplot(2,1,2)
#     # plt.imshow(a_im_ar,cmap = "gray")
#     # plt.show()
#     ID +=1


def random_dataset2():
    running = True
    ID = 0
    manual_set = []
    print(manual_set)
    manual_number_list = np.array([int(21),int(150),int(165),int(256),int(378),int(529),int(736),int(770),int(868),int(926),int(981),int(1082),int(1117),int(1223),int(1299),int(1321),int(1326),int(1435),int(1628)])
    while running:
        pic_num = manual_number_list[random.randint(1,18)]
        sec_num = random.randint(1,10)
        num_ar = [pic_num,sec_num]
        if num_ar in manual_set:
            print(f"Found dupe {num_ar}")
        else:
            manual_set.append(num_ar)
            print(f"{num_ar}")
            ID +=1
            mask = Image.open(f"Data Processed/Training/watershed/masks/Tape_B_2_-{pic_num}_{sec_num}.tif")
            im = Image.open(f"Data Processed/Training/watershed/images/Tape_B_2_-{pic_num}_{sec_num}.tif")
            mask.save(f"Data Processed/Training/dataset2/masks/Tape_B_2_{pic_num}_{sec_num}.tif")
            im.save(f"Data Processed/Training/dataset2/images/Tape_B_2_{pic_num}_{sec_num}.tif")
        if ID >=30:
            print(manual_set)
            running = False


def dim_checker():
    ID=0
    for filename in os.listdir("Data Processed/Training/manual/masks"):
        mask_im = Image.open(f"Data Processed/Training/manual/masks/{filename}")
        im_im = Image.open(f"Data Processed/Training/manual/images/{filename}")
        mask_ar = np.array(mask_im)
        im_ar = np.array(im_im)
        ID +=1
        print(ID)

        if np.shape(im_ar) != np.shape(mask_ar):
            print(f"{filename}, {np.shape(mask_ar),np.shape(im_ar)}")
# im = Image.open(f"Data Processed/Training/manual/images/Tape_B_2_5.tif")
# im_ar = np.array(im)
# im_ar = im_ar[:,2:]
# im = Image.fromarray(im_ar)
# im.save(f"Data Processed/Training/manual/images/Tape_B_2_5.tif")
print(len(os.listdir("Data Processed/Training/watershed/images")))
