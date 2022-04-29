import os

import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt


for i in range(1,21):
    file = open(f"Annotated/Tape_B_2_csv (watershed made GT)/Tape_B_2_{i}.csv")
    a = np.genfromtxt(file,delimiter = ",")
    print(a)
    a_strip = a[1:,1:]
    im = Image.fromarray(a_strip)
    im.save(f"Annotated/Tape_B_2_tif(watershed made GT)/Tape_B_2_{i}.tif")