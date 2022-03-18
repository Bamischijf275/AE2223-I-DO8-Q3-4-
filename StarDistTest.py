import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from stardist import random_label_cmap
from stardist.models import StarDist2D


# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize

lbl_cmap = random_label_cmap()


#img = test_image_nuclei_2d()

img = Image.open('/Users/ryan/PycharmProjects/scientificProject/TapeA_registration1stSlice.jpg')

# convert image object into array
img = np.asarray(img)

labels, _ = model.predict_instances(normalize(img))
np.savetxt("mask.csv", labels, delimiter=",", fmt="%.0f")


plt.figure(figsize=(8,8))
plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.axis('off')

plt.show()

