#  Copyright (c) 2022. Bart Lamiroy (Bart.Lamiroy@univ-reims.fr) and subsequent contributors
#  as per git commit history. All rights reserved.
#
#  La Muse, Leveraging Artificial Intelligence for Sparking Inspiration
#  https://hal.archives-ouvertes.fr/hal-03470467/
#
#  This code is licenced under the GNU LESSER GENERAL PUBLIC LICENSE
#  Version 3, 29 June 2007
#

import os

# os.chdir('./Mask_RCNN')
from glob import glob

from tensorflow.python.keras.backend import reset_uids

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()

from .compare_images import best_image
from .generate_segmented_pictures import get_segmented_mask
from ..Musesetup import *

from PIL import Image, ImageOps
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from .MaskRCNNModel import MaskRCNNModel

# @Todo find out why there are global variables and how to (maybe) get rid of them
# object_file_list = {}
object_image_list_nested = {}
object_image_list = []


def create_image_with_shapes(background_image: np.ndarray, painting: np.ndarray, r: dict, cursor) -> tuple:
    """
    Replace the objects on the background using shapes corresponding to the painting

    :param background_image:
    :param painting:
    :param r:
    :param cursor:
    :return:
    """
    # @TODO : find how to make images not looking blue

    nb_element = r['class_ids'].size
    real_value = None

    # dispach the value of the cursor between all elements
    if nb_element != 0:
        cursor = cursor / nb_element

    for i in range(nb_element):
        # print("Replace a ", MaskRCNNModel.class_names[r['class_ids'][i]])
        # @TODO: find out why real_value is reset to 0.0
        real_value = 0.0

        shape_mask = get_segmented_mask(painting, r, i)
        cropped_shape = shape_mask[r['rois'][i][0]: r['rois'][i][2], r['rois'][i][1]: r['rois'][i][3]]  # crop the image
        """fig, axs = plt.subplots(1, 2)
        fig.suptitle(str(i))
        axs[0].imshow(target_image)
        axs[0].set_title("image de base")
        axs[1].imshow(segment)
        axs[1].set_title("image modèle")
        plt.show()"""

        # Pick object that best fit the hole
        if cropped_shape is not None:
            # get the image with the best shape
            # @TODO get rid of this global 'object_image_list' variable
            replacement_shape, result, _ = best_image(cropped_shape, object_image_list, cursor)
            real_value += result
            """fig, axs = plt.subplots(1, 2)
            fig.suptitle(str(i))
            axs[0].imshow(target_image)
            axs[0].set_title("image de base")
            axs[1].imshow(result_image)
            axs[1].set_title("image modèle")
            plt.show()"""

            replacement_object = Image.fromarray(replacement_shape)  # convert to Image

            # Définition des dimensions et du placement du futur objet à coller
            original_object_bbox = (r['rois'][i][1], r['rois'][i][0], r['rois'][i][3], r['rois'][i][2])
            original_object_width = original_object_bbox[2] - original_object_bbox[0]
            original_object_height = original_object_bbox[3] - original_object_bbox[1]

            # Resizing replacement_object to original_object dimensions.
            replacement_object = replacement_object.resize((original_object_width, original_object_height),
                                                           Image.ANTIALIAS)

            # Paste replacement_object into background_image using alpha channel
            background_image.paste(replacement_object, (r['rois'][i][1], r['rois'][i][0]), replacement_object)
        else:
            print("Warning : None image")

    return background_image, real_value


def create_image_with_categories_and_shapes(background_image, painting, r, cursor):
    nb_element = r['class_ids'].size
    real_value = None
    # dispach the value of the cursor between all elements
    if nb_element != 0:
        cursor = cursor / nb_element

    for i in range(nb_element):

        current_class = MaskRCNNModel.class_names[r['class_ids'][i]]
        # print("Replace a ", current_class)
        real_value = 0.0

        source_mask = get_segmented_mask(painting, r, i)
        source_mask = source_mask[r['rois'][i][0]: r['rois'][i][2], r['rois'][i][1]: r['rois'][i][3]]  # crop the image

        # Pick object that best fits the hole
        if source_mask is not None:
            # get the image with the best shape
            result_image, result, _ = best_image(source_mask, object_image_list_nested[current_class], cursor)
            real_value += result

            replacement_object = Image.fromarray(result_image)  # convert to Image

            # Définition des dimensions et du placement du futur objet à coller
            original_object_bbox = (r['rois'][i][1], r['rois'][i][0], r['rois'][i][3], r['rois'][i][2])
            original_object_width = original_object_bbox[2] - original_object_bbox[0]
            original_object_height = original_object_bbox[3] - original_object_bbox[1]

            # Resizing replacement_object to original_object dimensions.
            replacement_object = replacement_object.resize((original_object_width, original_object_height),
                                                           Image.ANTIALIAS)

            # Paste replacement_object into background_image using alpha channel
            background_image.paste(replacement_object, (r['rois'][i][1], r['rois'][i][0]), replacement_object)
        else:
            print("Warning : None image")

    return background_image, real_value


# replace the objects on the background using categories corresponding to the painting
def create_image_with_categories(background_image, painting, r, cursor):
    for i in range(r['class_ids'].size):

        current_class = MaskRCNNModel.class_names[r['class_ids'][i]]
        # print("Replace a ", current_class)

        # get image in the painting (as done when generating images)
        # Pick a random object
        try:
            replacement_object = Image.fromarray(random.choice(object_image_list_nested[current_class]))
        except IndexError:
            print("Cannot find", current_class)  # , "in", path_to_substitute_objects, "for", painting_name)
            break
        # Définition des dimensions et du placement du futur objet à coller
        original_object_bbox = (r['rois'][i][1], r['rois'][i][0], r['rois'][i][3], r['rois'][i][2])
        original_object_width = original_object_bbox[2] - original_object_bbox[0]
        original_object_height = original_object_bbox[3] - original_object_bbox[1]

        # Resizing replacement_object to original_object dimensions.
        replacement_object = replacement_object.resize((original_object_width, original_object_height),
                                                       Image.ANTIALIAS)

        # Paste replacement_object into background_image using alpha channel
        background_image.paste(replacement_object, (r['rois'][i][1], r['rois'][i][0]), replacement_object)
    return background_image, 0


def create_case_study(path_to_paintings: str, path_to_substitute_objects: str,
                      path_to_background_images: str,
                      path_to_results: str,
                      nb_paintings: int = 3) -> None:
    """
    :param path_to_paintings:
    :param path_to_substitute_objects:
    :param path_to_background_images:
    :param path_to_results:
    :param nb_paintings:
    :return:
    """
    path_to_results += '/'

    cursor = 0
    cursor_step = 0
    if nb_paintings != 1:
        cursor_step = 20 / (nb_paintings - 1)  # cursor values in [0,20]

    '''
    list_of_methods = [create_image_with_shapes] * nb_paintings + [create_image_with_categories_and_shapes] * nb_paintings + [
        create_image_with_categories] * nb_paintings
    method_names = ["shapes", "shapes and categories", "categories"]
    '''
    list_of_methods = [create_image_with_categories_and_shapes] * nb_paintings
    method_names = ["categories-shapes"]

    if not os.path.exists(path_to_results):
        os.mkdir(path_to_results)

    # Run the model on the painting.
    model = MaskRCNNModel().model

    painting_file_list = [y for x in [glob(path_to_paintings + '/*.%s' % ext) for ext in image_extensions] for y in x]

    # List of available background images
    background_file_list = \
        [y for x in [glob(path_to_background_images + '/*.%s' % ext) for ext in image_extensions] for y in x]

    if len(object_image_list) == 0 or len(object_image_list_nested) == 0:
        print("Updating objects...")
        # List of candidate replacement objects
        for obj in MaskRCNNModel.class_names:
            # object_file_list[obj] = [y for x in [glob(path_to_substitute_objects + '/%s/*.%s' % (obj, ext))
            # for ext in image_extensions] for y in x]
            file_list = [y for x in [glob(path_to_substitute_objects + '/%s/*.%s' % (obj, ext))
                                     for ext in image_extensions] for y in x]
            object_image_list_nested[obj] = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in file_list]
            [object_image_list.append(img) for img in object_image_list_nested[obj]]
        # for nested_list in object_file_list.values():
        # for i in nested_list:
        # object_image_list.append(cv2.imread(i, cv2.IMREAD_UNCHANGED))

    for painting_filename in painting_file_list:
        print("\nPainting : ", painting_filename)
        painting_name = os.path.basename(painting_filename)
        painting = load_img(painting_filename)
        painting = img_to_array(painting)
        painting_width, painting_height = painting.shape[1], painting.shape[0]

        # Extract significant items from painting
        results = model.detect([painting], verbose=0, probability_criteria=0.7)
        r = results[0]

        cursor = 0
        j = 0

        # Generate a number of altered forms of painting for each technic
        for technic in list_of_methods:
            print("Painting number : ", j)

            # Pick a random background image
            try:
                background_image_name = random.choice(background_file_list)
            except IndexError:
                print(path_to_background_images + " is empty, taking initial image instead")
                background_image_name = painting_filename

            background_image = Image.open(background_image_name)
            # Resize the background image with the size of painting.
            background_image = background_image.resize((painting_width, painting_height), Image.ANTIALIAS)
            background_image = background_image.convert("RGBA")

            background_image, real_value = technic(background_image, painting, r, cursor)
            if real_value is None:
                real_value = -1.0
            # background_image, real_value = create_image_with_shapes(background_image, painting, r, cursor)

            # Save background_image.
            file_saved = path_to_results + painting_name + "-method=" + method_names[
                j // nb_paintings] + "-value=" + '%.3f' % real_value + '.png'
            background_image = background_image.convert("RGB")
            background_image.save(file_saved)
            file_saved = path_to_results + painting_name + "-method=" + method_names[
                j // nb_paintings] + "-value=" + '%.3f' % real_value + '.pgm'
            background_image = ImageOps.grayscale(background_image)
            background_image.save(file_saved)

            print("Real value obtained : ", real_value)
            cursor += cursor_step  # to have different result for an image
            j += 1


if __name__ == "__main__":
    painting_dir = default_painting_folder
    background_dir = default_background_folder
    output_dir = './NewRésultats2/'

    create_case_study(painting_dir, default_substitute_folder, background_dir, output_dir)
