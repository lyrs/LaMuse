"""
    This program computes hue histograms with various levels of smoothing from a random image
"""

from glob import glob
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

import cv2

background_dir = './BaseImages/paysages décors fonds/'
test_image = './Paintings/800px-Piero_della_Francesca_042.jpg'
image_extensions = ["jpg", "gif", "png", "tga"]

background_image_list = [y for x in [glob(background_dir + '/*.%s' % ext) for ext in image_extensions] for y in x]

test_image = random.choice(background_image_list)
print(test_image)

# OpenCV approach (HSV)

image_open_cv = cv2.imread(test_image)
image_HSV = cv2.cvtColor(image_open_cv, cv2.COLOR_BGR2HSV)

# Create histogram of hues in image_HSV
# 180 is max Hue value
dict_HSV = {k: 0 for k in range(180)}

for width in range(image_HSV.shape[0]):
    for height in range(image_HSV.shape[1]):
        dict_HSV[(image_HSV[width, height, 0])] += 1

data = [dict_HSV[i] for i in range(180)]

figure = plt.figure()

a = figure.add_subplot(1, 4, 1)
a.set_title('Plot')
a.plot([i for i in range(180)], data, color='red')
img = Image.open(test_image)
b = figure.add_subplot(1, 4, 4)
b.set_title('Image')
b.imshow(img)
# c = figure.add_subplot(1,4,2)
# c.set_title('Plot With Filter')

# w = savgol_filter(data, 105, 2)
# c.plot([i for i in range(180)], w , color = 'green')
d = figure.add_subplot(1, 4, 3)
d.set_title('Savgol filter')
y = savgol_filter(data, 47, 2)
d.plot([i for i in range(180)], y, color='green')

# LOESS Filter : Smoothing data with local regression.

# Moyennes glissantes

moyennes_glissantes = [0, 0]

for i in range(2, 178):
    moyennes_glissantes.append((data[i - 2] + data[i - 1] + data[i] + data[i + 1] + data[i + 2]) / 5)

moyennes_glissantes.append(0)
moyennes_glissantes.append(0)


def abs_list(alist):
    return [np.abs(x) for x in alist]


e = figure.add_subplot(1, 4, 2)
e.set_title('Moyennes glissantes with savgol filter')
e.plot([i for i in range(180)], abs_list(savgol_filter(moyennes_glissantes, 47, 2)), color='blue')

plt.show()
