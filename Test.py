from glob import glob
import os
import cv2

from glob import glob

from random import randint
import matplotlib.pyplot as plt

from .tools.MaskRCNNModel import MaskRCNNModel
from .tools.barycentre import barycentre, visu_Barycentre

import pkg_resources


import PySimpleGUI as sg
from .tools.generate_segmented_pictures import generate_images
from .tools.create_original_case_study import create_case_study
from .tools.fast_style_transfer import save_image
from .tools.compare_images import best_image, getOrientationAndScale

segmentation_suffix = "_objets"

default_image_folder = './LaMuse/BaseImages'
default_background_folder = None
default_painting_folder = './LaMuse/Paintings'
default_interpretation_folder = './Interpretations'

mask_rcnn_config_file = os.path.dirname(__file__) + '/mask_rcnn_coco.h5'


if __name__ == "__main__":
    # example give the best image found in BaseImage_objects
    cursor = 0.0
    path_objects_to_replace = "./LaMuse/BaseImages_objets"
    image_extensions = ["jpg", "gif", "png", "tga"]

    # All possible images
    object_file_list = []
    for obj in MaskRCNNModel.class_names:
        object_file_list.extend([y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                               for ext in image_extensions] for y in x])
    
    # test barycenter
    """r = randint(0, len(object_file_list) - 1)
    target_file = object_file_list[r]
    target_image = cv2.imread(target_file, cv2.IMREAD_UNCHANGED)
    a,b = barycentre(target_image)
    print(a,b)
    image_bar = visu_Barycentre(target_image,a,b)
    plt.imshow(image_bar)
    plt.show()"""
    
    
    print("nombre total d'images : ", len(object_file_list))
    r = randint(0,len(object_file_list)-1)
    target_file = object_file_list[r]

    object_file_list.remove(target_file) # not comparing target with itself

    # images , IMREAD_UNCHANGED to make sure alpha channel is loaded
    object_image_list = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in object_file_list]
    target_image = cv2.imread(target_file, cv2.IMREAD_UNCHANGED)

    image, result = best_image(target_image, object_image_list, cursor)

    fig, axs = plt.subplots(1,2)
    fig.suptitle("Différence de bizaritude obtenue par rapport a la valeur demandée : " + str(result) + "\nValeur cible : " + str(cursor))

    axs[0].imshow(target_image)
    axs[1].imshow(image)
    plt.show()

    """
    img = blackAndWhitePNG(target_image)
    print(img)
    plt.imshow(img)
    plt.show()
    """