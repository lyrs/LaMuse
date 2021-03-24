import cv2
import numpy as np
import matplotlib.pyplot as plt

from .barycentre import barycentre

THRESHOLD_RATIO = 10

# https://docs.opencv.org/master/d5/d45/tutorial_py_contours_more_functions.html
# return the best image according to target and the value mesuring the shape difference
def best_image(target, image_list:list, cursor:float):
    best = None # positive value
    best_indice = -1 # indice of the best image
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
def getOrientationAndScale(contour):
    # get rotated rectangle from outer contour
    largerContour = contour[0]
    for i in contour:
        if i.shape >largerContour.shape :
            largerContour = i
    #print("Contour choisi : ", largerContour.shape)
    rect = cv2.minAreaRect(largerContour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get angle from rotated rectangle
    angle = rect[-1]

    # from https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    # rect : (x,y), (width, height), angle
    #print("height : ", rect[1][1], " width : ", rect[1][0])
    if rect[1][0] > rect[1][1] : # width < height
        angle += 90

    #print("Angle de l'image : ", angle,"deg")
    return angle

def applyOrientation(contour1, target, contour2, image):
    angle1 = getOrientationAndScale(contour1)
    angle2 = getOrientationAndScale(contour2)
    angleDiff = angle1 - angle2 # vérifier l'intervalle
    #print("Rotation de :", angleDiff)
    # to if the image is + 180 or not

    image = rotate_bound(image, angleDiff) # first angle

    # on vérifie les barycentres pour savoir si on ajoute une rotation de 180 degrés :
    angleDiff = 0
    b_img = barycentre(image)
    b_target = barycentre(target)
    if (b_img[0] < image.shape[0]/2 and b_target[0] > target.shape[0]/2) or (b_img[0]> image.shape[0]/2 and b_target[0]<target.shape[0]/2) : # cas gauche / droite
        angleDiff =180
        #print ("+180")
    elif (b_img[1] < image.shape[1]/2 and b_target[1] > target.shape[1]/2) or (b_img[1]> image.shape[1]/2 and b_target[1]<target.shape[1]/2) :
        angleDiff =180
        #print ("+180")

    image = rotate_bound(image, angleDiff) # first angle

    """ test de l
    rotated_2 = rotate_bound(image, angleDiff+180) # +180 version

    img_gray_1 = cv2.cvtColor(blackAndWhitePNG(rotated_1),cv2.COLOR_BGR2GRAY) # gray scale to compare
    img_gray_2 = cv2.cvtColor(blackAndWhitePNG(rotated_2),cv2.COLOR_BGR2GRAY)

    norm_1 = np.linalg.norm(gray - img_gray_1)  # compute the norme to know how much is the difference
    norm_2 = np.linalg.norm(gray - img_gray_2)

    if norm_1 < norm_2 : # choose the angle that has the minimum différence
        return rotated_1
    """

    """fig, axs = plt.subplots(1,2)
    axs[0].imshow(scaled)
    axs[0].set_title("rotated")
    axs[1].imshow(image)
    axs[1].set_title("origin")
    plt.show()"""
    return image

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






