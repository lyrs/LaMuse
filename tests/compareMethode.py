import numpy as np
from glob import glob
import os
import time
import cv2
from glob import glob

from random import randint
import matplotlib.pyplot as plt

from ..tools.MaskRCNNModel import MaskRCNNModel
from ..tools.compare_images import transparancy_mask_to_BW

from ..setup import *

path_objects_to_replace = default_substitute_folder

def bizzaritude_method(target: np.ndarray, image_list: list, cursor: float, methode):
    best = None  # positive value
    bizaritude_list = []
    img_gray = cv2.cvtColor(transparancy_mask_to_BW(target), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    target_contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, methode)  # contour of the target
    for i in range(len(image_list)):
        img_gray = cv2.cvtColor(transparancy_mask_to_BW(image_list[i]), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, methode)  # contour of the image
        if len(contours) == 0 or len(target_contour) == 0:
            print("error")
            continue
        ret = cv2.matchShapes(contours[0], target_contour[0], cv2.CONTOURS_MATCH_I1, 0.0)  # mesure the difference
        if ret < 100:
            bizaritude_list.append(ret)
    return bizaritude_list  # , best


def tri_compte_liste(lst: list) -> tuple:
    lst.sort()
    occurrence = []
    der = -1
    for i in range(len(lst)):
        if lst[i] > der:
            occurrence.append(lst.count(lst[i]))
            der = lst[i]
    lst = list(set(lst))
    return lst, occurrence


def compare_toutes_images(methode):
    cursor = 0.0

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

    liste_donne = bizzaritude_method(target_image, object_image_list, cursor, methode)
    return liste_donne


def comparaison_methode():
    t = []
    t1 = time.perf_counter()
    liste_simple, occ_simple = tri_compte_liste(compare_toutes_images(cv2.CHAIN_APPROX_SIMPLE))
    t2 = time.perf_counter()
    t.append((t2 - t1))
    print(len(liste_simple), len(occ_simple))

    t1 = time.perf_counter()
    liste_none, occ_none = tri_compte_liste(compare_toutes_images(cv2.CHAIN_APPROX_NONE))
    t2 = time.perf_counter()
    t.append((t2 - t1))
    print(len(liste_none), len(occ_none))

    t1 = time.perf_counter()
    liste_l1, occ_l1 = tri_compte_liste(compare_toutes_images(cv2.CHAIN_APPROX_TC89_L1))
    t2 = time.perf_counter()
    t.append((t2 - t1))
    print(len(liste_l1), len(occ_l1))

    t1 = time.perf_counter()
    liste_kcos, occ_kcos = tri_compte_liste(compare_toutes_images(cv2.CHAIN_APPROX_TC89_KCOS))
    t2 = time.perf_counter()
    t.append((t2 - t1))
    print(len(liste_kcos), len(occ_kcos))

    names = ['CHAIN_APPROX_SIMPLE', 'CHAIN_APPROX_NONE', 'CHAIN_APPROX_TC89_L1',
             'CHAIN_APPROX_TC89_KCOS']  # sample names
    print(t)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(liste_simple, occ_simple, s=130, c='green', marker='+')

    axs[0, 0].set_title(f'CHAIN_APPROX_SIMPLE t= {t[0]} 10^3 nanosecondes')
    axs[0, 1].scatter(liste_none, occ_none, s=130, c='blue', marker='+')
    axs[0, 1].set_title(f'CHAIN_APPROX_NONE t= {t[1]}')
    axs[1, 0].scatter(liste_l1, occ_l1, s=50, c='red', marker='+')
    axs[1, 0].set_title(f'CHAIN_APPROX_TC89_L1 t= {t[2]}')
    axs[1, 1].scatter(liste_kcos, occ_kcos, s=50, c='black', marker='+')
    axs[1, 1].set_title(f'CHAIN_APPROX_TC89_KCOS, t= {t[3]}')

    for ax in axs.flat:
        ax.set(xlabel='Mesure de la bizaritude', ylabel='Ocurence')

    plt.show()

    # plt.xlabel('Names')
    # plt.ylabel('Probability')

    return


comparaison_methode()
