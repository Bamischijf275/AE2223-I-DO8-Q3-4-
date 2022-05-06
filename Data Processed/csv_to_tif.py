import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random

def ID_renamer(ar):
    ID = 0
    for j in np.unique(ar):
        if j !=0:
            ID +=1
            ar[ar==j] = ID
    return ar
#
# for name in os.listdir("Data Processed/Watershed/Training"):
#     print(f"{name}")
#     if name != str("TIF"):
#         ar = np.genfromtxt(f"Data Processed/Watershed/Training/{name}",delimiter=",")
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
#             ar_top_im.save(f'Data Processed/Training/dataset4/masks/{filename}_{top}.tif')
#             ar_bot_im.save(f'Data Processed/Training/dataset4/masks/{filename}_{bot}.tif')
#             top +=1
#             bot +=1

def array_splitter(ar):
    ar_split = np.array_split(ar,5,axis=1)
    top = 1
    bot = 6
    ar = [[],[],[],[],[],[],[],[],[],[]]
    for j in range(0,5):
        ar_topbot = ar_split[j]
        ar_topbot = np.array_split(ar_topbot,2,axis=0)
        ar_top = ar_topbot[0].astype("uint16")
        ar_bot = ar_topbot[1].astype("uint16")
        ar_top = ID_renamer(ar_top)
        ar_bot = ID_renamer(ar_bot)
        top +=1
        bot +=1
        ar[top] = ar_top
        ar[bot] = ar_bot
        print(ar)
        return ar



# #
# ID=0
# for name in os.listdir("Data Processed/AI results/dataset4/images"):
#     a_m = Image.open(f'Data Processed/AI results/dataset4/masks/{name}')
#     a_im = Image.open(f'Data Processed/AI results/dataset4/images/{name}')
#     a_m_ar = np.array(a_m)
#     a_im_ar = np.array(a_im)
#     if np.shape(a_m_ar) != np.shape(a_im_ar):
#         print(np.shape(a_m_ar),np.shape(a_im_ar))
#         print(f"Placeholder {ID},{name}")
#     fig = plt.figure(figsize=(5,6))
#     fig.add_subplot(1,5,1)
#     plt.imshow(a_m_ar,cmap = "gray")
#     fig.add_subplot(2,1,2)
#     plt.imshow(a_im_ar,cmap = "gray")
#     plt.show()
#     input(f"{name}")
#     ID +=1

def add_im_mask():
    for name in os.listdir("Data Processed/Watershed/Training"):
        if name != str("TIF"):
            name = name.replace(".csv", "")
            im = Image.open(f"Data/Tape_B/Tape_B_2_JPG/{name}.jpg")
            ar = np.array(im)
            ar_split = np.array_split(ar, 5, axis=1)
            top = 1
            bot = 6
            for j in range(0,5):
                ar_topbot = ar_split[j]
                ar_topbot = np.array_split(ar_topbot,2,axis=0)
                ar_top = ar_topbot[0].astype("uint8")
                ar_bot = ar_topbot[1].astype("uint8")
                ar_top = ID_renamer(ar_top)
                ar_bot = ID_renamer(ar_bot)
                ar_top_im = Image.fromarray(ar_top,)
                ar_bot_im = Image.fromarray(ar_bot)
                ar_top_im.save(f'Data Processed/Training/dataset4/images/{name}_{top}.tif')
                ar_bot_im.save(f'Data Processed/Training/dataset4/images/{name}_{bot}.tif')
                top +=1
                bot +=1


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
    for filename in os.listdir("Data Processed/Training/dataset4/masks"):
        mask_im = Image.open(f"Data Processed/Training/dataset4/masks/{filename}")
        im_im = Image.open(f"Data Processed/Training/dataset4/images/{filename}")
        mask_ar = np.array(mask_im)
        im_ar = np.array(im_im)
        if np.shape(im_ar) != np.shape(mask_ar):
            print(f"{filename}, {np.shape(mask_ar),np.shape(im_ar)}")
            ID += 1
    if ID == 0:
        print("all dimensions are correct")

def color_image_maker(dataset):
    for filename in os.listdir(f"Data Processed/AI Results/{dataset}/masks"):
        mask = Image.open(f"Data Processed/AI Results/{dataset}/masks/{filename}")
        mask_ar = np.array(mask)
        filename = filename.replace(".tif", "")
        color_ar = np.zeros((np.shape(mask_ar)[1],np.shape(mask_ar)[0],3))
        print(np.shape(color_ar))
        color_store = []
        mask_ar_unique = np.unique(mask_ar)[1:]
        running = True
        ID=0
        while running:
            R = random.randint(1,255)
            G = random.randint(1,255)
            B = random.randint(1,255)
            RGB_ar = [R,G,B]
            if RGB_ar not in color_store:
                color_store.append(RGB_ar)
                print(RGB_ar,color_store)
                color_ar[np.where(mask_ar == mask_ar_unique[ID])] = RGB_ar
                ID+=1
            if ID >= len(mask_ar_unique):
                running = False
        print(color_ar)
        color_im = Image.fromarray(color_ar,"RGB")
        color_im.save(f"Data Processed/AI Results/{dataset}/pic_mask/{filename}.pdf")
        exit()
color_image_maker(str("dataset2"))
# im = Image.open(f"Data Processed/Training/manual/images/Tape_B_2_5.tif")
# im_ar = np.array(im)
# im_ar = im_ar[:,2:]
# im = Image.fromarray(im_ar)
# im.save(f"Data Processed/Training/manual/images/Tape_B_2_5.tif")
# print(len(os.listdir("Data Processed/Watershed/Training")))
# ar = []
# running = True
# ID = 0
# while running:
#     number = random.randint(1,len(os.listdir("Data Processed/Training/watershed/images")))
#     list = os.listdir("Data Processed/Training/watershed/images")
#     if number in ar:
#         print("Dupe")
#     else:
#         im = Image.open(f"Data Processed/Training/watershed/images/{list[number]}")
#         mas = Image.open(f"Data Processed/Training/watershed/masks/{list[number]}")
#         im.save(f'Data Processed/Training/dataset3/images/{list[number]}')
#         mas.save(f'Data Processed/Training/dataset3/masks/{list[number]}')
#         ar.append(number)
#         ID +=1
#     if ID >= 100:
#         running = False
# print(len(os.listdir("Data Processed/Training/dataset3/images")))
#
