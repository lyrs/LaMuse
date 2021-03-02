import cv2
import numpy as np
from glob import glob
import os
import cv2
from glob import glob

from random import randint
import matplotlib.pyplot as plt

from .tools.MaskRCNNModel import MaskRCNNModel

from .tools.compare_images import blackAndWhitePNG

# todo : comparer le temps mis pour voir si les approximations ont bien une influence

segmentation_suffix = "_objets"

default_image_folder = './LaMuse/BaseImages'
default_background_folder = None
default_painting_folder = './LaMuse/Paintings'
default_interpretation_folder = './Interpretations'

mask_rcnn_config_file = os.path.dirname(__file__) + '/mask_rcnn_coco.h5'

def best_image(target, image_list:list, cursor:float, approx = cv2.CHAIN_APPROX_NONE, method = cv2.CONTOURS_MATCH_I1):
    best = None # positive value
    listeBizzaritude = []
    img_gray = cv2.cvtColor(blackAndWhitePNG(target),cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray, 127, 255,0)
    targetContour,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, approx) # contour of the target
    for i in range(len(image_list)):
        img_gray = cv2.cvtColor(blackAndWhitePNG(image_list[i]),cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray, 127, 255,0)
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, method) # contour of the image
        if (len(contours) == 0 or len(targetContour) == 0):
            print("error")
            continue
        ret = cv2.matchShapes(contours[0],targetContour[0], method ,0.0) # mesure the difference
        diff = abs(ret-cursor)
        if diff > 50 : diff = 100
        listeBizzaritude.append(diff)
        if (best == None or diff<best): # find best
            best = diff
    return listeBizzaritude 

def triCompteListe(liste):
    liste.sort()
    
    occurence = []
    der = -1
    for i in range(len(liste)):
        if liste[i] > der :
            occurence.append(liste.count(liste[i]))
            der = liste[i]
    liste = list(set(liste))
    return (liste,occurence)

def compareToutesImages(methode):
    cursor = 0.0

    print("nombre total d'images : ", len(object_file_list))
    r = randint(0, len(object_file_list) - 1)
    target_file = object_file_list[r]

    # images , IMREAD_UNCHANGED to make sure alpha channel is loaded
    object_image_list = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in object_file_list]
    target_image = cv2.imread(target_file, cv2.IMREAD_UNCHANGED)

    listeDonne = best_image(target_image, object_image_list, cursor, methode)
    #print(listeDonne)
    return listeDonne

def comparaisonMethode():
    listapprox = [cv2.CHAIN_APPROX_SIMPLE,cv2.CHAIN_APPROX_NONE,cv2.CHAIN_APPROX_TC89_L1,cv2.CHAIN_APPROX_TC89_KCOS]
    for i in listapprox:
        liste, occ = triCompteListe(compareToutesImages(i))
        plt.plot(liste, occ)
    plt.show()
    return

path_objects_to_replace = "./LaMuse/BaseImages_objets"
image_extensions = ["jpg", "gif", "png", "tga"]
object_file_list = []
for obj in MaskRCNNModel.class_names:
    object_file_list.extend([y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                            for ext in image_extensions] for y in x])
print(triCompteListe([5.0,1,1.5,1,15.2,5,2,24,5,5,3,5]))
comparaisonMethode()
