import cv2
import numpy as np
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
from .tools.compare_images import best_image, blackAndWhitePNG

segmentation_suffix = "_objets"

default_image_folder = './LaMuse/BaseImages'
default_background_folder = None
default_painting_folder = './LaMuse/Paintings'
default_interpretation_folder = './Interpretations'

mask_rcnn_config_file = os.path.dirname(__file__) + '/mask_rcnn_coco.h5'

def bizzaritudeMethode(target, image_list:list, cursor:float, methode):
    best = None # positive value
    listeBizzaritude = []
    img_gray = cv2.cvtColor(blackAndWhitePNG(target),cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray, 127, 255,0)
    targetContour,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, methode) # contour of the target
    for i in range(len(image_list)):
        img_gray = cv2.cvtColor(blackAndWhitePNG(image_list[i]),cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray, 127, 255,0)
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, methode) # contour of the image
        if (len(contours) == 0 or len(targetContour) == 0):
            print("error")
            continue
        ret = cv2.matchShapes(contours[0],targetContour[0], cv2.CONTOURS_MATCH_I1 ,0.0) # mesure the difference
        diff = abs(ret-cursor)
        listeBizzaritude.append(diff)
        if (best == None or diff<best): # find best
            best = diff
    return listeBizzaritude #, best


def triCompteListe(liste):
    liste.sort()
    occurence = []
    der = 0
    for i in range(len(liste)):
        if liste[i] > der :
            occurence.append(liste.count(liste[i]))
            der = liste[i]
    liste = set(liste)
    return (liste,occurence)

def compareToutesImages(methode):
    cursor = 0.0
    path_objects_to_replace = "./LaMuse/BaseImages_objets"
    image_extensions = ["jpg", "gif", "png", "tga"]

    # All possible images
    object_file_list = []
    for obj in MaskRCNNModel.class_names:
        object_file_list.extend([y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                             for ext in image_extensions] for y in x])

    print("nombre total d'images : ", len(object_file_list))
    r = randint(0, len(object_file_list) - 1)
    target_file = object_file_list[r]

    object_file_list.remove(target_file)  # not comparing target with itself

    # images , IMREAD_UNCHANGED to make sure alpha channel is loaded
    object_image_list = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in object_file_list]
    target_image = cv2.imread(target_file, cv2.IMREAD_UNCHANGED)

    listeDonne = bizzaritudeMethode(target_image, object_image_list, cursor, methode)
    return listeDonne

def comparaisonMethode():
    listeSimple, occSimple = triCompteListe(compareToutesImages(cv2.CHAIN_APPROX_SIMPLE))
    print(len(listeSimple),len(occSimple))
    listeNone, occNone = triCompteListe(compareToutesImages(cv2.CHAIN_APPROX_NONE))
    print(len(listeNone),len(occNone))
    listeL1, occL1 = triCompteListe(compareToutesImages(cv2.CHAIN_APPROX_TC89_L1))
    print(len(listeL1),len(occL1))
    listeKcos, occKcos = triCompteListe(compareToutesImages(cv2.CHAIN_APPROX_TC89_KCOS))
    print(len(listeKcos),len(occKcos))


    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(listeSimple, occSimple)
    axs[0, 0].set_title('CHAIN_APPROX_SIMPLE')
    plt.show()
    axs[0, 1].plot(listeNone, occNone, 'tab:orange')
    axs[0, 1].set_title('CHAIN_APPROX_NONE')
    axs[1, 0].plot(listeL1, occL1, 'tab:green')
    axs[1, 0].set_title('CHAIN_APPROX_TC89_L1')
    axs[1, 1].plot(listeKcos, occKcos, 'tab:red')
    axs[1, 1].set_title('CHAIN_APPROX_TC89_KCOS')

    plt.show()
    return

comparaisonMethode()
