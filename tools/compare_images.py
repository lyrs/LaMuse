#  Copyright (c) 2022. Bart Lamiroy (Bart.Lamiroy@univ-reims.fr) and subsequent contributors
#  as per git commit history. All rights reserved.
#
#  La Muse, Leveraging Artificial Intelligence for Sparking Inspiration
#  https://hal.archives-ouvertes.fr/hal-03470467/
#
#  This code is licenced under the GNU LESSER GENERAL PUBLIC LICENSE
#  Version 3, 29 June 2007
#
import random
from math import fabs, log

import cv2
import numpy as np

from .barycentre import barycentre


def best_image(source_image: np.ndarray, candidate_image_list: list, cursor: float,
               threshold_ratio: float = 3.0) -> tuple:
    """
    from https://docs.opencv.org/master/d5/d45/tutorial_py_contours_more_functions.html

    return the best image according to target and the value measuring the shape difference
    Images are supposed to be template images with transparent background

    :param source_image: image for which to find the best replacement
    :param candidate_image_list: list of images to find the replacement in
    :param cursor:
    :param threshold_ratio:
    :return:
    """

    best = 0  # positive value
    nb_ratio_correct = 0
    best_index = -1  # index of the best image
    img_gray = cv2.cvtColor(transparancy_mask_to_BW(source_image), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    source_contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_NONE)  # contour of the source_image
    if len(source_contour) == 0:
        try:
            raise ValueError
        except ValueError as error:
            error.message = error.message + " source_image image has no contours"
            raise

    # @Todo this is an ugly hack !! All contours should be handled, not just the first one
    best_contour = source_contour[0].copy()
    best_candidate = source_image

    log_best_ratio = 10000
    best_ratio_image = source_image

    matched_shapes = {}

    found_best = False
    for candidate_image in candidate_image_list:

        if candidate_image.size == 0:
            continue

        current_ratio = source_image.size / candidate_image.size
        log_current_ratio = fabs(log(current_ratio))
        if log_current_ratio < log_best_ratio:
            log_best_ratio = log_current_ratio
            best_ratio_image = candidate_image

        if current_ratio < 1 / threshold_ratio or \
                current_ratio > threshold_ratio:
            continue

        nb_ratio_correct += 1
        img_gray_temp = cv2.cvtColor(transparancy_mask_to_BW(candidate_image), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray_temp, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # contour of the image
        if len(contours) == 0:
            print("Found image without contours")
            continue
        ret = cv2.matchShapes(contours[0], source_contour[0], cv2.CONTOURS_MATCH_I1, 0.0)  # measure the difference
        # @Todo this is an ugly hack !! All contours should be handled, not just the first one
        diff = abs(ret - cursor)
        matched_shapes[diff] = (candidate_image, contours[0].copy())

        # =====
        # This is legacy code kept for reference ... of no further use
        if diff < best or not found_best:  # find best
            best = diff
            found_best = True
            # @Todo this is an ugly hack !! All contours should be handled, not just the first one
            best_contour = contours[0].copy()
            best_candidate = candidate_image
        # =====

    nb_matched_shapes = len(matched_shapes)

    if nb_matched_shapes > 0:
        random_scope = int(max(nb_matched_shapes * 0.05, 10))
        random_scope = min(nb_matched_shapes, random_scope)

        sorted_keys = sorted(matched_shapes.keys())
        sorted_scope = sorted_keys[:random_scope]
        random_index = random.choice(sorted_scope)

        random_item = matched_shapes[random_index]
        best_contour = random_item[1]
        best_candidate = random_item[0]
        best = random_index
    else:
        best_candidate = best_ratio_image
        img_gray_temp = cv2.cvtColor(transparancy_mask_to_BW(best_candidate), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray_temp, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best_contour = contours[0].copy()

    # rotate the image to have the same orientation
    res = apply_orientation(source_contour, source_image, best_contour, best_candidate)
    # return candidate_image_list[best_indice], best, nbRatioCorrect
    return res, best, nb_ratio_correct


def transparancy_mask_to_BW(img: np.ndarray) -> np.ndarray:
    """
    Transforms a transparency mask image into a black (background) and white (foreground) mask image
    :param img: original mask image with transparent layer
    :return: copy of mask image with foreground set to white and background to black.
    """
    res_img = img.copy()
    mask = res_img[:, :, 3] == 0  # transparent areas
    res_img[mask] = [0, 0, 0, 255]  # transparent -> black (background)
    mask = np.logical_not(mask)
    res_img[mask] = [255, 255, 255, 255]  # others -> white (object)
    return res_img


def get_orientation(contours: list) -> float:
    """
    Computes and returns the orientation angle (in degrees) of the largest component in the provided contour list
    (from https://stackoverflow.com/questions/58632469/how-to-find-the-orientation-of-an-object-shape-python-opencv)

    :param contours: list of contours
    :return: orientation angle of the minimal surrounding rectangle (taken at its largest side)
    """
    # get rotated rectangle from outer contour
    larger_contour = contours[0]
    for i in contours:
        if i.shape > larger_contour.shape:
            larger_contour = i
    # print("Contour choisi : ", larger_contour.shape)

    # rect elements : [(x,y), (width, height), angle]
    rect = cv2.minAreaRect(larger_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get angle from rotated rectangle
    angle = rect[-1]
    # print("Angle de l'image sans traitement : ", angle,"deg")

    # calculate the angle based on the bigger side
    if rect[1][0] < rect[1][1]:  # width < height
        angle -= 90

    # print("Angle de l'image : ", angle,"deg")
    return angle


def apply_orientation(target_contour, target, image_contour, image):
    angle1 = get_orientation(target_contour)
    angle2 = get_orientation(image_contour)
    angle_diff = angle1 - angle2
    # print("Rotation de :", angle_diff)

    # turn the image in the same direction as the source_image
    rotated = center_rotate_image(image, angle_diff)

    # Check with the barycenter whether the image is flipped or not
    angle_diff = 0

    b_img = barycentre(image)
    b_target = barycentre(target)

    # if the barycenter is not in the same side (left / right -> x axis) of both image
    if ((b_img[0] < image.shape[0] / 2 and b_target[0] > target.shape[0] / 2)
            or (b_img[0] > image.shape[0] / 2 and b_target[0] < target.shape[0] / 2)):
        angle_diff = 180
    # if the barycenter is not in the same side (top / bottom -> y axis) of both image
    elif ((b_img[1] < image.shape[1] / 2 and b_target[1] > target.shape[1] / 2)
          or (b_img[1] > image.shape[1] / 2 and b_target[1] < target.shape[1] / 2)):
        angle_diff = 180

    """# rectangle that define the center of the image
    min_image_x = image.shape[0]/2 - margin * image.shape[0]
    max_image_x = image.shape[0]/2 + margin * image.shape[0]
    min_image_y = image.shape[1]/2 - margin * image.shape[1]
    max_image_y = image.shape[1]/2 + margin * image.shape[1]
    # rectangle that define the center of the source_image
    min_target_x = source_image.shape[0]/2 - margin * source_image.shape[0]
    max_target_x = source_image.shape[0]/2 + margin * source_image.shape[0]
    min_target_y = source_image.shape[1]/2 - margin * source_image.shape[1]
    max_target_y = source_image.shape[1]/2 + margin * source_image.shape[1]

    # if the barycenter is not in the same side (left / right -> x axis) of both image
    if ((b_img[0] < min_image_x and b_target[0] > max_target_x) 
        or (b_img[0]> max_image_x and b_target[0] < min_target_x)) :
        angle_diff =180
        print ("Cas numéro 1 : +180")
    
    # if the barycenter is not in the same side (top / bottom -> y axis) of both image
    if ((b_img[0] < min_image_y and b_target[0] > max_target_y) 
        or (b_img[0]> max_image_y and b_target[0] < min_target_y)) :
        angle_diff =180
        print ("Cas numéro 2 : +180")"""

    # print("Ajout d'un angle de ", angle_diff, "deg")
    corrected = center_rotate_image(rotated, angle_diff)  # second rotation if necessary

    return corrected


def center_rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

    Rotates the given input image over its center coordinates (computed as the center of its bounding box) with
    the angle given as argument

    :param image: input image to rotate
    :param angle: rotation angle
    :return: new image rotated by the required angle
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # (cX, cY) = bar
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    transformation_matrix = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(transformation_matrix[0, 0])
    sin = np.abs(transformation_matrix[0, 1])
    # compute the new bounding dimensions of the image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    transformation_matrix[0, 2] += (new_w / 2) - cX
    transformation_matrix[1, 2] += (new_h / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, transformation_matrix, (new_w, new_h))
