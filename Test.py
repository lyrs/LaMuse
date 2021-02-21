from glob import glob
import os
import cv2

from glob import glob

from random import randint
import matplotlib.pyplot as plt

from .tools.MaskRCNNModel import MaskRCNNModel

import pkg_resources


import PySimpleGUI as sg
from .tools.generate_segmented_pictures import generate_images
from .tools.create_original_case_study import create_case_study
from .tools.fast_style_transfer import save_image
from .tools.compare_images import best_image

segmentation_suffix = "_objets"

default_image_folder = './LaMuse/BaseImages'
default_background_folder = None
default_painting_folder = './LaMuse/Paintings'
default_interpretation_folder = './Interpretations'

mask_rcnn_config_file = os.path.dirname(__file__) + '/mask_rcnn_coco.h5'


if __name__ == "__main__":
    # example give the best image found in BaseImage_objects
    cursor = 20
    path_objects_to_replace = "./LaMuse/BaseImages_objets"
    image_extensions = ["jpg", "gif", "png", "tga"]

    # All possible images
    object_file_list = []
    for obj in MaskRCNNModel.class_names:
        object_file_list.extend([y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                             for ext in image_extensions] for y in x])
    
    print(len(object_file_list))
    r = randint(0,len(object_file_list)-1)
    target_file = object_file_list[r]

    # images
    object_image_list = [cv2.imread(i) for i in object_file_list]
    target_image = cv2.imread(target_file)

    image, result = best_image(target_image, object_image_list, cursor)

    print("Résultat de la bizaritude obtenue : ", result)

    plt.figure(1)
    plt.imshow(target_image)
    plt.figure(2)
    plt.imshow(image)
    plt.show()