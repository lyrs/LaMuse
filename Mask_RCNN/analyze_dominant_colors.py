import PIL
from PIL import Image
import os
import random
import numpy as np
import tkinter as tk 
import cv2
import matplotlib.pyplot as plt

list_backgrounds = os.listdir('../BaseImages/paysages décors fonds/')
if '.DS_Store' in list_backgrounds:
    list_backgrounds.remove('.DS_Store')
image_path = random.choice(list_backgrounds)


## OpenCV approach (HSV)

image_open_cv = cv2.imread('../../../800px-Piero_della_Francesca_042.jpg')
image_HSV = cv2.cvtColor(image_open_cv, cv2.COLOR_BGR2HSV)
list_hues = []

for width in range(image_HSV.shape[0]):
    for height in range(image_HSV.shape[1]):
        list_hues.append(image_HSV[width, height, 0])

dict_HSV = {k: 0 for k in range(180)}

for hue in list_hues:
    dict_HSV[hue] += 1

data = [dict_HSV[i] for i in range(180)]

figure = plt.figure()

a = figure.add_subplot(1,4,1)
a.set_title('Plot')
a.plot([i for i in range(180)], data , color = 'red')
img = Image.open('../../../800px-Piero_della_Francesca_042.jpg')
b = figure.add_subplot(1,4,4)
b.set_title('Image')
b.imshow(img)
#c = figure.add_subplot(1,4,2)
#c.set_title('Plot With Filter')
from scipy.signal import savgol_filter
#w = savgol_filter(data, 105, 2)
#c.plot([i for i in range(180)], w , color = 'green')
d = figure.add_subplot(1,4,3)
d.set_title('Savgol filter')
y = savgol_filter(data, 47, 2)
d.plot([i for i in range(180)], y , color = 'green')

## LOESS Filter : Smoothing data with local regression.

## Moyennes glissantes 

moyennes_glissantes = [0,0]

for i in range(2,178):

    moyennes_glissantes.append((data[i-2] + data[i-1] + data[i] + data[i+1] + data[i+2])/5)

moyennes_glissantes.append(0)
moyennes_glissantes.append(0)

def absList(l):
    return [np.abs(x) for x in l]

e = figure.add_subplot(1,4,2)
e.set_title('Moyennes glissantes with savgol filter')
e.plot([i for i in range(180)], absList(savgol_filter(moyennes_glissantes, 47, 2))  , color = 'blue')

plt.show()





















"""
for image_path in list_backgrounds:
    ## PIL Simple approach.
    
    image = Image.open('../BaseImages/paysages décors fonds/' + image_path)

    width, height = image.size

    R,G,B = 0,0,0

    for current_width in range(width):
        for current_height in range(height):
            R += float(image.getpixel((current_width, current_height))[0])
            G += float(image.getpixel((current_width, current_height))[1])
            B += float(image.getpixel((current_width, current_height))[2])
    R /= width*height
    G /= height*width
    B /= width*height 

    def _from_rgb(rgb):
        return "#%02x%02x%02x" % rgb
    """
    






"""
CORDA
"""

"""
fen = tk.Tk()

fen.configure(bg=_from_rgb((int(R),int(G),int(B))))

image.show()

fen.mainloop()
"""


