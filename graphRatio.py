import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from .tools.barycentre import barycentre
from .tools.MaskRCNNModel import MaskRCNNModel
from glob import glob


def comparaisonGraph(ratioMax: int):
    x=[]
    y=[]
    ratioCorrect = []

    path_objects_to_replace = "./LaMuse/BaseImages_objets"
    image_extensions = ["jpg", "gif", "png", "tga"]

    # All possible images
    object_file_list = []
    for obj in MaskRCNNModel.class_names:
        object_file_list.extend([y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                             for ext in image_extensions] for y in x])
    object_image_list = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in object_file_list]
    shuffle(object_image_list)
    empty = np.empty((0,0))
    #image_list_unchanged = object_image_list.copy()
    for i in range(2,ratioMax):
        x.append(i)
        res_add = 0
        print("Ratio : ", i)
        k=0
        for j in range(len(object_file_list)-1,0,-200):
            target_image = object_image_list[j]
            object_image_list[j] = empty
            res, ratio = best_image(target_image, object_image_list,0.0, i)[1:]
            res_add = res_add + res
            object_image_list[j] = target_image
            k+=1
        y.append(res_add/k)
        ratioCorrect.append(ratio)
        #object_image_list = image_list_unchanged.copy()
    plt.figure(1)
    plt.title("Bizaritude moyenne en fonction du ratio chosit")
    plt.plot(x,y)

    plt.figure(2)
    plt.title("Nombre d'image correspondant au ratio")
    plt.plot(x,ratioCorrect)
    plt.show()

# https://docs.opencv.org/master/d5/d45/tutorial_py_contours_more_functions.html
# return the best image according to target and the value mesuring the shape difference
def best_image(target, image_list: list, cursor: float, THRESHOLD_RATIO: int):
    best = None  # positive value
    nbRatioCorrect = 0; 
    #best_contour = None
    best_indice = -1  # indice of the best image
    img_gray = cv2.cvtColor(blackAndWhitePNG(target), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    targetContour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)  # contour of the target

    for i in range(len(image_list)):
        if image_list[i].size == 0:
            #print("oui")
            continue
        if (target.size / image_list[i].size) < 1 / THRESHOLD_RATIO or (
                target.size / image_list[i].size) > THRESHOLD_RATIO:
            # print("images too different")
            continue
        nbRatioCorrect+=1
        img_gray_temp = cv2.cvtColor(blackAndWhitePNG(image_list[i]), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray_temp, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # contour of the image
        if (len(contours) == 0 or len(targetContour) == 0):
            print("error")
            continue
        ret = cv2.matchShapes(contours[0], targetContour[0], cv2.CONTOURS_MATCH_I1, 0.0)  # mesure the difference
        diff = abs(ret - cursor)
        if (best == None or diff < best):  # find best
            best = diff
            best_indice = i
            #best_contour = contours.copy()
    #if (best_contour == None):
        #return(None)
    #res = applyOrientation(targetContour, target, best_contour, image_list[best_indice])
    return image_list[best_indice], best, nbRatioCorrect


def blackAndWhitePNG(img):
    resImg = img.copy()
    mask = resImg[:, :, 3] == 0  # transparent areas
    resImg[mask] = [0, 0, 0, 255]  # transparent -> black (background)
    mask = np.logical_not(mask)
    resImg[mask] = [255, 255, 255, 255]  # others -> white (object)
    return resImg

comparaisonGraph(30)