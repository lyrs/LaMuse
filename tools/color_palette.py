#  Copyright (c) 2022. Bart Lamiroy (Bart.Lamiroy@univ-reims.fr) and subsequent contributors
#  as per git commit history. All rights reserved.
#
#  La Muse, Leveraging Artificial Intelligence for Sparking Inspiration
#  https://hal.archives-ouvertes.fr/hal-03470467/
#
#  This code is licenced under the GNU LESSER GENERAL PUBLIC LICENSE
#  Version 3, 29 June 2007

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from .rgb_to_colorname import Ntc

# inspired from https://pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
def color_quantify(image: np.array, nb: int) -> np.array:
    # grab image width and height
    (h, w) = image.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = nb)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)

    return quant

# code adapted from https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python
def bincount_app(a: np.array, nb: int):
    # Not reducing image colours to three significant bits per channel since image supposedly already
    # reduced.
    #a = a & int("11100000", 2)
    cv2.imwrite("final_image.jpg",a)

    a2D = a.reshape(-1,a.shape[-1])

    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)

    #return np.unravel_index(np.bincount(a1D).argmax(), col_range)
    return np.unravel_index(np.bincount(a1D).argpartition(-nb)[-nb:], col_range)

def int_array_to_hex_string(a: list) -> str:
    str_list = [ "{0:#0{1}x}".format(i,4)[2:] for i in a]
    return '#'+''.join(str_list)


def get_color_names(image: np.array, nb_colors: int = 5):
    test_image = color_quantify(image, nb_colors)
    main_colors = [int_array_to_hex_string(i) for i in zip(*bincount_app(test_image, nb_colors))]
    naming = Ntc()
    main_color_names = [ naming.name(i)[1] for i in main_colors]

    return main_color_names




