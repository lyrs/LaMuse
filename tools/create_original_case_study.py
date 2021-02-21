import os

# os.chdir('./Mask_RCNN')
from glob import glob

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()

import random
import cv2

import PIL
from PIL import Image
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from .MaskRCNNModel import MaskRCNNModel


# draw an image with detected objects

def draw_image_with_boxes(filename: str, boxes_list: list) -> None:
    """
    :param filename:
    :param boxes_list:
    :return:
    """
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for box in boxes_list:
        # get coordinates
        y1, x1, y2, x2 = box
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=5)
        # draw the box
        ax.add_patch(rect)
    # show the plot
    pyplot.show()


def create_case_study(path_to_paintings: str, path_objects_to_replace: str,
                      path_to_background_images: str,
                      path_to_results: str,
                      nb_paintings: int = 5) -> None:
    """
    :param path_to_paintings:
    :param path_objects_to_replace:
    :param path_to_background_images:
    :param path_to_results:
    :param nb_paintings:
    :return:
    """
    path_to_results += '/'

    if not os.path.exists(path_to_results):
        os.mkdir(path_to_results)

    # Run the model on the painting.
    # @Todo find a way to invoke MaskRCNNModel only once
    model = MaskRCNNModel().model

    image_extensions = ["jpg", "gif", "png", "tga"]
    painting_file_list = [y for x in [glob(path_to_paintings + '/*.%s' % ext) for ext in image_extensions] for y in x]

    # List of available background images
    background_file_list = \
        [y for x in [glob(path_to_background_images + '/*.%s' % ext) for ext in image_extensions] for y in x]

    object_file_list = {}
    # List of candidate replacement objects
    for obj in MaskRCNNModel.class_names:
        object_file_list[obj] = [y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                             for ext in image_extensions] for y in x]
    for painting_filename in painting_file_list:

        painting_name = os.path.basename(painting_filename)
        painting = load_img(painting_filename)
        painting = img_to_array(painting)
        painting_width, painting_height = painting.shape[1], painting.shape[0]

        # Extract significant items from painting
        results = model.detect([painting], verbose=0, probability_criteria=0.7)
        r = results[0]

        # Generate a number of altered forms of painting
        NUMBER_OF_TRIES = nb_paintings

        for j in range(NUMBER_OF_TRIES):
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

            # file_saved = path_to_results + painting_name + str(j) + '.png'
            # background_image.save(file_saved)

            for i in range(r['class_ids'].size):
                ##
                # On doit enregistrer l'image à toutes les itérations car la fonction
                # putpixel ne fonctionne qu'après avoir sauvegardé les changements apportés
                # Ainsi on charge l'image temporaire, dans les cas où l'itération courante n'est pas la première
                ##
                current_class = MaskRCNNModel.class_names[r['class_ids'][i]]

                # if i != 0:
                #    background_image = Image.open(path_to_results + 'temp.png')

                # Pick a random object
                try:
                    replacement_object = Image.open(random.choice(object_file_list[current_class]))
                except IndexError:
                    print("Cannot find", current_class, "in", path_objects_to_replace, "for", painting_name)
                    break

                # Définition des dimensions et du placement du futur objet à coller
                original_object_bbox = (r['rois'][i][1], r['rois'][i][0], r['rois'][i][3], r['rois'][i][2])
                original_object_width = original_object_bbox[2] - original_object_bbox[0]
                original_object_height = original_object_bbox[3] - original_object_bbox[1]

                # Resizing replacement_object to original_object dimensions.
                # @Todo: respect aspect ratio of replacement object
                replacement_object = replacement_object.resize((original_object_width, original_object_height),
                                                               Image.ANTIALIAS)

                # Paste replacement_object into background_image using alpha channel
                background_image.paste(replacement_object, (r['rois'][i][1], r['rois'][i][0]), replacement_object)

                # Save background_image.
                # if i == r['class_ids'].size - 1:
                #    file_saved = path_to_results + painting_name + str(j) + '.png'
                #    background_image = background_image.convert("RGB")
                #    background_image.save(file_saved)
                # else:
                #    background_image.save(path_to_results + 'temp.png')

            file_saved = path_to_results + painting_name + str(j) + '.png'
            background_image = background_image.convert("RGB")
            background_image.save(file_saved)


if __name__ == "__main__":
    painting_dir = './Paintings/'
    background_dir = './BaseImages/paysages décors fonds'
    output_dir = './NewRésultats2/'

    create_case_study(painting_dir, './CroppedAndVectorizedImages/animaux/', background_dir, output_dir)
