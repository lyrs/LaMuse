import cv2
import numpy as np


# https://docs.opencv.org/master/d5/d45/tutorial_py_contours_more_functions.html
# return the best image according to target and the value mesuring the shape difference
def best_image(target, image_list:list, cursor:float, approx = cv2.CHAIN_APPROX_NONE, method = cv2.CONTOURS_MATCH_I1):
    best = None # positive value
    best_indice = 0 # indice of the best image
    img_gray = cv2.cvtColor(blackAndWhitePNG(target),cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray, 127, 255,0)
    targetContour,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, approx) # contour of the target
    for i in range(len(image_list)):
        img_gray = cv2.cvtColor(blackAndWhitePNG(image_list[i]),cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray, 127, 255,0)
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, approx) # contour of the image
        if (len(contours) == 0 or len(targetContour) == 0):
            print("error")
            continue
        ret = cv2.matchShapes(contours[0],targetContour[0], method ,0.0) # mesure the difference
        diff = abs(ret-cursor)
        if (best == None or diff<best): # find best
            best = diff
            best_indice = i
    return image_list[best_indice], best 


def blackAndWhitePNG(img):
    resImg = img.copy()
    mask = resImg[:,:,3] == 0  # transparent areas
    resImg[mask] = [0, 0, 0, 255]  # transparent -> black (background)
    mask = np.logical_not(mask)
    resImg[mask] = [255, 255, 255, 255]  # others -> white (object)
    return resImg






