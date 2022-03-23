import numpy as np
from stardist import random_label_cmap
from stardist.models import StarDist2D
from PIL import Image

# prints a list of available models
StarDist2D.from_pretrained()

# creates a pretrained model
model = StarDist2D.from_pretrained('2D_versatile_fluo')

from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt

lbl_cmap = random_label_cmap()


img = Image.open('TapeA_registration.jpg')

# convert image object into array
img = np.asarray(img)

labels, _ = model.predict_instances(normalize(img))

plt.figure(figsize=(8,8))
#plt.imshow(img if img.ndim==2 else img[...,0], clim=(0,1), cmap='gray')
plt.imshow(labels, cmap=lbl_cmap, alpha=0.5)
plt.axis('off')

plt.show()

