import numpy as np
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

#img = test_image_nuclei_2d()
img = Image.open('/Users/ryan/PycharmProjects/scientificProject/TapeA_registration1stSlice.jpg')

# convert image object into array
img = np.asarray(img)
print(img)
labels, _ = model.predict_instances(normalize(img))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")

plt.show()
