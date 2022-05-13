import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def ID_renamer(ar):
    ID = 0
    for j in np.unique(ar):
        if j !=0:
            ID +=1
            ar[ar==j] = ID
    return ar
def Tabe_B_toCSV():
    for name in os.listdir("Data/Tape_B/Masks_tif"):
        ar_im = Image.open(f"Data/Tape_B/Masks_tif/{name}")
        ar_ar = np.array(ar_im)
        filename = name.replace(".tif", "")
        np.savetxt(f"Data Processed/Annotated/mask_csv/{filename}.csv",ar_ar, delimiter=",")

# for name in os.listdir("Data/Tape_B/Masks_tif"):
#     print(f"{name}")
#     if name != str("TIF"):
#         ar_im = Image.open(f"Data/Tape_B/Masks_tif/{name}")
#         ar_ar = np.array(ar_im)
#         #ar = np.genfromtxt(f"Data/Tape_B/Masks_tif/{name}",delimiter=",")
#         filename = name.replace(".tif","")
#         #ar = ar[1:,1:]
#         ar_split = np.array_split(ar_ar,5,axis=1)
#         top = 1
#         bot = 6
#         for j in range(0,5):
#             ar_topbot = ar_split[j]
#             ar_topbot = np.array_split(ar_topbot,2,axis=0)
#             ar_top = ar_topbot[0].astype("uint16")
#             ar_bot = ar_topbot[1].astype("uint16")
#             ar_top = ID_renamer(ar_top)
#             ar_bot = ID_renamer(ar_bot)
#             np.savetxt(f"Data Processed/Annotated/mask_csv/{filename}_{top}.csv",ar_top, delimiter=",")
#             np.savetxt(f"Data Processed/Annotated/mask_csv/{filename}_{bot}.csv", ar_bot, delimiter=",")
#             #ar_top_im = Image.fromarray(ar_top,)
#             #ar_bot_im = Image.fromarray(ar_bot)
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
# for name in os.listdir("Data Processed/AI results/dataset4_ver1/images"):
#     a_m = Image.open(f'Data Processed/AI results/dataset4_ver1/masks/{name}')
#     a_im = Image.open(f'Data Processed/AI results/dataset4_ver1/images/{name}')
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
                ar_top_im.save(f'Data Processed/Training/dataset4_ver1/images/{name}_{top}.tif')
                ar_bot_im.save(f'Data Processed/Training/dataset4_ver1/images/{name}_{bot}.tif')
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
            mask = Image.open(f"Data Processed/Training/control algorithm/masks/Tape_B_2_-{pic_num}_{sec_num}.tif")
            im = Image.open(f"Data Processed/Training/control algorithm/images/Tape_B_2_-{pic_num}_{sec_num}.tif")
            mask.save(f"Data Processed/Training/dataset2`_ver1/masks/Tape_B_2_{pic_num}_{sec_num}.tif")
            im.save(f"Data Processed/Training/dataset2`_ver1/images/Tape_B_2_{pic_num}_{sec_num}.tif")
        if ID >=30:
            print(manual_set)
            running = False


def dim_checker():
    ID=0
    for filename in os.listdir("Data Processed/Training/dataset4_ver1/masks"):
        mask_im = Image.open(f"Data Processed/Training/dataset4_ver1/masks/{filename}")
        im_im = Image.open(f"Data Processed/Training/dataset4_ver1/images/{filename}")
        mask_ar = np.array(mask_im)
        im_ar = np.array(im_im)
        if np.shape(im_ar) != np.shape(mask_ar):
            print(f"{filename}, {np.shape(mask_ar),np.shape(im_ar)}")
            ID += 1
    if ID == 0:
        print("all dimensions are correct")

def get_mask(dataset,filename):
    mask = Image.open(f"Data Processed/AI results/{dataset}/masks/{filename}")
    mask_ar = np.array(mask)
    name = filename.replace(".tif","")
    tab20c = cm.get_cmap("tab20c", 256)
    cmap_cus = tab20c(np.linspace(0, 1, len(np.unique(mask_ar))))
    white = np.array([0, 0, 0, 1])
    cmap_cus[:1, :] = white
    newcmp = ListedColormap(cmap_cus)
    return mask_ar,dataset,newcmp

def color_image_maker(filename):
    mask_array = []
    dataset_names = []
    fig = plt.figure(constrained_layout = True)
    ax = fig.add_gridspec(3,2)
    ax1 = fig.add_subplot(ax[0,1])
    ax_11 = fig.add_subplot(ax[0,0])
    ax2 = fig.add_subplot(ax[1,0])
    ax3 = fig.add_subplot(ax[1,1])
    ax4 = fig.add_subplot(ax[2,0])
    ax5 = fig.add_subplot(ax[2,1])
    #set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.4,
                        top=0.9,
                        wspace=0.1,
                        hspace=0.3)
    #Ground thruth image
    GT_im = Image.open(f"Data Processed/Annotated/tif/{filename}")
    GT_ar = np.array(GT_im)
    tab20c = cm.get_cmap("tab20c", 256)
    cmap_cus = tab20c(np.linspace(0, 1, len(np.unique(GT_ar))))
    white = np.array([0, 0, 0, 1])
    cmap_cus[:1, :] = white
    newcmp = ListedColormap(cmap_cus)
    ax1.set_title("GT")
    ax1.imshow(GT_ar,cmap = newcmp,interpolation='nearest')
    ax1.axis("off")
    ax_11.set_title("Image")
    ax_11.imshow(Image.open(f"Data Processed/AI results/dataset1/images/{filename}"),cmap="gray")
    ax_11.axis("off")
    ID = 0

    for dataset in os.listdir("Data Processed/AI results"):
        print(dataset,ID)
        mask_ar,dataset,newcmp = get_mask(dataset,filename)

        if ID == 0:
            ax2.set_title(f"StarDist1")
            ax2.imshow(mask_ar,cmap= newcmp,interpolation='nearest')
            ax2.axis("off")
        if ID == 1:
            ax3.set_title(f"StarDist2")
            ax3.imshow(mask_ar,cmap= newcmp,interpolation='nearest')
            ax3.axis("off")
        if ID == 2:
            ax4.set_title(f"StarDist3")
            ax4.imshow(mask_ar,cmap= newcmp,interpolation='nearest')
            ax4.axis("off")
        if ID == 4:
            ax5.set_title(f"Control")
            filename = filename.replace(".tif","")
            print(f"Data Processed/Annotated/Watershed/Extras/{filename}_step_7_Out.png")
            ax5.imshow(mpimg.imread(f"Data Processed/Watershed/Extras/{filename}_step_7_Out.png"),interpolation="nearest")
            ax5.axis("off")
        ID +=1
    plt.savefig(f"Data processed/AI comparison pictures/{filename}.pdf",bbox_inches="tight",pad_inches=0)
    plt.show()
color_image_maker(str('Tape_B_7_3.tif'))
def csv_to_mask():
    for name in os.listdir(f"Data Processed/Annotated"):
        if name == str("Tape_B_1_4.csv") or name == str("Tape_B_1_7.csv") or name == str("Tape_B_2_3.csv") or name == str("Tape_B_3_8.csv") or name == str("Tape_B_5_7.csv") or name == str("Tape_B_6_6.csv") or name == str("Tape_B_7_3.csv") or name == str("Tape_B_8_5.csv") or name == str("Tape_B_8_9.csv") or name == str("Tape_B_11_6.csv"):
            ar = np.genfromtxt(f"Data Processed/Annotated/{name}", delimiter=",")
            ar = ar[1:,1:]
            name = name.replace(".csv","")
            image = Image.fromarray(ar)
            print(name)
            image.save(f"Data Processed/AI results/watershed/{name}.tif")

def mask_to_csv(dataset):
    for name in os.listdir(f"Data Processed/AI results/{dataset}/masks"):
        im = Image.open(f"Data Processed/AI results/{dataset}/masks/{name}")
        filename = name.replace(".tif","")
        im_ar = np.array(im)
        np.savetxt(f"Data Processed/AI results/{dataset}/mask_csv/{filename}.csv",im_ar,delimiter = ",")
# for filename in os.listdir("Data Processed/AI results/dataset1/images"):
#     color_image_maker(filename)

# im = Image.open(f"Data Processed/Training/dataset1/images/Tape_B_2_5.tif")
# im_ar = np.array(im)
# im_ar = im_ar[:,2:]
# im = Image.fromarray(im_ar)
# im.save(f"Data Processed/Training/dataset1/images/Tape_B_2_5.tif")
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
#         im.save(f'Data Processed/Training/dataset3_ver1/images/{list[number]}')
#         mas.save(f'Data Processed/Training/dataset3_ver1/masks/{list[number]}')
#         ar.append(number)
#         ID +=1
#     if ID >= 100:
#         running = False
# print(len(os.listdir("Data Processed/Training/dataset3_ver1/images")))
#
