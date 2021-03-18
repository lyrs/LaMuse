import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape


# https://docs.opencv.org/master/d5/d45/tutorial_py_contours_more_functions.html
# return the best image according to target and the value mesuring the shape difference
def best_image(target, image_list:list, cursor:float):
    best = None # positive value
    best_indice = -1 # indice of the best image
    img_gray = cv2.cvtColor(blackAndWhitePNG(target),cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray, 127, 255,0)
    targetContour,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # contour of the target
    
    for i in range(len(image_list)):
        img_gray = cv2.cvtColor(blackAndWhitePNG(image_list[i]),cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(img_gray, 127, 255,0)
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

    res = applyOrientation(targetContour, best_contour, image_list[best_indice])
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
    rect = cv2.minAreaRect(contour[0])
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
    print("height : ", rect[1][1], " width : ", rect[1][0])
    if rect[1][0] > rect[1][1] : # width < height
        angle += 90

    scale = 1 # not calculated for the moment
    print("Angle de l'image : ", angle,"deg")
    return angle, scale

def applyOrientation(contour1, contour2, image):
    angle1, scale1 = getOrientationAndScale(contour1)
    angle2, scale2 = getOrientationAndScale(contour2)
    angleDiff = angle1 - angle2 # vérifier l'intervalle
    print("Rotation de :", angleDiff)
    scaleDiff = scale1/scale2
    rotated = rotate_bound(image, angleDiff)
    scaled = rotated #cv2.resize(rotated, (int(image.shape[1] * scaleDiff),int(image.shape[0] * scaleDiff)))    
    """fig, axs = plt.subplots(1,2)
    axs[0].imshow(scaled)
    axs[0].set_title("rotated")
    axs[1].imshow(image)
    axs[1].set_title("origin")
    plt.show()"""
    return rotated

#idées pour l'angle : 
# https://stackoverflow.com/questions/24073127/opencvs-rotatedrect-angle-does-not-provide-enough-information en c++
#

# from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
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






