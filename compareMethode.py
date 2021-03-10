import numpy as np
from glob import glob
import os
import time
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


def bizzaritudeMethode(target, image_list: list, cursor: float, methode):
    best = None  # positive value
    listeBizzaritude = []
    img_gray = cv2.cvtColor(blackAndWhitePNG(target), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    targetContour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, methode)  # contour of the target
    for i in range(len(image_list)):
        img_gray = cv2.cvtColor(blackAndWhitePNG(image_list[i]), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, methode)  # contour of the image
        if (len(contours) == 0 or len(targetContour) == 0):
            print("error")
            continue
        ret = cv2.matchShapes(contours[0], targetContour[0], cv2.CONTOURS_MATCH_I1, 0.0)  # mesure the difference
        if (ret < 100):
            listeBizzaritude.append(ret)
    return listeBizzaritude  # , best


def triCompteListe(liste):
    liste.sort()
    occurence = []
    der = 0
    for i in range(len(liste)):
        if liste[i] > der:
            occurence.append(liste.count(liste[i]))
            der = liste[i]
    liste = list(set(liste))
    return (liste, occurence)


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
    t=[]
    t1 = time.perf_counter()
    listeSimple, occSimple = triCompteListe(compareToutesImages(cv2.CHAIN_APPROX_SIMPLE))
    t2 = time.perf_counter()
    t.append((t2 - t1))
    print(len(listeSimple), len(occSimple))

    t1 = time.perf_counter()
    listeNone, occNone = triCompteListe(compareToutesImages(cv2.CHAIN_APPROX_NONE))
    t2 = time.perf_counter()
    t.append((t2 - t1))
    print(len(listeNone), len(occNone))

    t1 = time.perf_counter()
    listeL1, occL1 = triCompteListe(compareToutesImages(cv2.CHAIN_APPROX_TC89_L1))
    t2 = time.perf_counter()
    t.append((t2 - t1))
    print(len(listeL1), len(occL1))

    t1 = time.perf_counter()
    listeKcos, occKcos = triCompteListe(compareToutesImages(cv2.CHAIN_APPROX_TC89_KCOS))
    t2 = time.perf_counter()
    t.append((t2 - t1))
    print(len(listeKcos), len(occKcos))

    names = ['CHAIN_APPROX_SIMPLE', 'CHAIN_APPROX_NONE', 'CHAIN_APPROX_TC89_L1','CHAIN_APPROX_TC89_KCOS']  # sample names
    print(t)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(listeSimple, occSimple, s=130, c='green', marker='+')

    axs[0, 0].set_title('CHAIN_APPROX_SIMPLE t= ' + t(0) + '10^3 nanoseconde')
    axs[0, 1].scatter(listeNone, occNone, s=130, c='blue', marker='+')
    axs[0, 1].set_title('CHAIN_APPROX_NONE t= ' + t(1))
    axs[1, 0].scatter(listeL1, occL1, s=50, c='red', marker='+')
    axs[1, 0].set_title('CHAIN_APPROX_TC89_L1 t= ' + t(2))
    axs[1, 1].scatter(listeKcos, occKcos, s=50, c='black', marker='+')
    axs[1, 1].set_title('CHAIN_APPROX_TC89_KCOS, t= ' + t(3))

    for ax in axs.flat:
        ax.set(xlabel='Mesure de la bizzaritude', ylabel='Ocurence')

    plt.show()

    # plt.xlabel('Names')
    # plt.ylabel('Probability')

    return


comparaisonMethode()
