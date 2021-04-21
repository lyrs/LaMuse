import cv2
import numpy as np
import matplotlib.pyplot as plt

from .barycentre import barycentre

THRESHOLD_RATIO = 10

# https://docs.opencv.org/master/d5/d45/tutorial_py_contours_more_functions.html
# return the best image according to target and the value mesuring the shape difference
def best_image(target:np.ndarray, image_list:list, cursor:float):
    #print ("Nombre d'images : ", len(image_list))
    best = None # positive value
    best_indice = -1 # index of the best image
    img_gray = cv2.cvtColor(blackAndWhitePNG(target),cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray, 127, 255,0)
    targetContour,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # contour of the target
    
    for i in range(len(image_list)):
        if (target.size / image_list[i].size) < 1/THRESHOLD_RATIO or (target.size / image_list[i].size) > THRESHOLD_RATIO:
            #print("images too different")
            continue
        img_gray_temp = cv2.cvtColor(blackAndWhitePNG(image_list[i]),cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray_temp, 127, 255,0)
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # contour of the image
        if (len(contours) == 0 or len(targetContour) == 0):
            print("error")
            continue
        ret = cv2.matchShapes(contours[0],targetContour[0], cv2.CONTOURS_MATCH_I1 ,0.0) # mesure the difference
        diff = abs(ret-cursor)
        if (best == None or diff<best): # find best
            best = diff
            best_indice = i
            best_contour = contours.copy()

    # rotate the image to have the same orientation
    res = applyOrientation(targetContour, target, best_contour, image_list[best_indice])
    return res, best 


def blackAndWhitePNG(img):
    resImg = img.copy()
    mask = resImg[:,:,3] == 0  # transparent areas
    resImg[mask] = [0, 0, 0, 255]  # transparent -> black (background)
    mask = np.logical_not(mask)
    resImg[mask] = [255, 255, 255, 255]  # others -> white (object)
    return resImg

# from https://stackoverflow.com/questions/58632469/how-to-find-the-orientation-of-an-object-shape-python-opencv
def getOrientation(contour):
    # get rotated rectangle from outer contour
    largerContour = contour[0]
    for i in contour:
        if i.shape > largerContour.shape :
            largerContour = i
    #print("Contour choisi : ", largerContour.shape)

    # rect elements : [(x,y), (width, height), angle]
    rect = cv2.minAreaRect(largerContour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get angle from rotated rectangle
    angle = rect[-1]
    print("Angle de l'image sans traitement : ", angle,"deg")
    
    # calculate the angle based on the bigger side
    if rect[1][0] < rect[1][1] : # width < height
        angle -= 90

    print("Angle de l'image : ", angle,"deg")
    return angle

def applyOrientation(target_contour, target, image_contour, image):
    angle1 = getOrientation(target_contour)
    angle2 = getOrientation(image_contour)
    angleDiff = angle1 - angle2
    print("Rotation de :", angleDiff)

    # turn the image in the same direction as the target
    rotated = rotate_bound(image, angleDiff) 

    # Check with the barycenter if the image is flipped
    angleDiff = 0
    
    b_img = barycentre(image)
    b_target = barycentre(target)
    if (b_img[0] < image.shape[0]/2 and b_target[0] > target.shape[0]/2) or (b_img[0]> image.shape[0]/2 and b_target[0]<target.shape[0]/2) : # cas gauche / droite
        angleDiff =180
        #print ("+180")
    elif (b_img[1] < image.shape[1]/2 and b_target[1] > target.shape[1]/2) or (b_img[1]> image.shape[1]/2 and b_target[1]<target.shape[1]/2) :
        angleDiff =180
        #print ("+180")

    print("Ajout d'un angle de ", angleDiff, "deg")
    corrected = rotate_bound(rotated, angleDiff) # second rotation if necessary

    return corrected

#idées pour l'angle : 
# https://stackoverflow.com/questions/24073127/opencvs-rotatedrect-angle-does-not-provide-enough-information en c++
#

# from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):#, bar):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    #(cX, cY) = bar
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))






