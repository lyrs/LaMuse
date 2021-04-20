import os

# os.chdir('./Mask_RCNN')
from glob import glob

from tensorflow.python.keras.backend import reset_uids

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()

from .compare_images import best_image
from .generate_segmented_pictures import getSegment

from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from .MaskRCNNModel import MaskRCNNModel

object_file_list = {}
object_image_list = []

# draw an image with detected objects

def draw_image_with_boxes(filename: str, boxes_list: list) -> None:
    """
    :param filename:
    :param boxes_list:
    :return:
    """
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
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
    plt.show()

# replace the objects on the background using shapes corresponding to the painting
def createImageWithShapes(background_image, painting, r, cursor):
    for i in range(r['class_ids'].size):
        print("Replace a ", MaskRCNNModel.class_names[r['class_ids'][i]])       
        #current_class = MaskRCNNModel.class_names[r['class_ids'][i]]

        # if i != 0:
        #    background_image = Image.open(path_to_results + 'temp.png')

        target_image = getSegment(painting, r, i)
        #plt.imshow(target_image)
        #plt.show()
        realValue = 0
        # Pick object that best fit the hole
        if not target_image is None:
            # get the image with the best shape
            result_image, result = best_image(target_image, object_image_list, cursor) 
            realValue+=result
            #plt.imshow(result_image)
            #plt.show()
            replacement_object = Image.fromarray(result_image) # convert to Image            

            # Définition des dimensions et du placement du futur objet à coller
            original_object_bbox = (r['rois'][i][1], r['rois'][i][0], r['rois'][i][3], r['rois'][i][2])
            original_object_width = original_object_bbox[2] - original_object_bbox[0]
            original_object_height = original_object_bbox[3] - original_object_bbox[1]

            # Resizing replacement_object to original_object dimensions.
            replacement_object = replacement_object.resize((original_object_width, original_object_height),
                                                        Image.ANTIALIAS)

            # Paste replacement_object into background_image using alpha channel
            background_image.paste(replacement_object, (r['rois'][i][1], r['rois'][i][0]), replacement_object)
        else :
            print("Warning : None image")

    return background_image, realValue

# replace the objects on the background using catégories corresponding to the painting
def createImageWithCategories(background_image, painting, r, cursor):
    for i in range(r['class_ids'].size):
               
        current_class = MaskRCNNModel.class_names[r['class_ids'][i]]
        print("Replace a ", current_class)

        #if i != 0:
        #    background_image = Image.open(path_to_results + 'temp.png')

        # get image in the painting (as done when generating images)
        # Pick a random object
        replacement_object = Image.open(random.choice(object_file_list[current_class]))

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


def create_case_study(path_to_paintings: str, path_objects_to_replace: str,
                      path_to_background_images: str,
                      path_to_results: str,
                      nb_paintings: int = 1) -> None:
    """
    :param path_to_paintings:
    :param path_objects_to_replace:
    :param path_to_background_images:
    :param path_to_results:
    :param nb_paintings:
    :return:
    """
    path_to_results += '/'

    cursor = 0
    cursor_diff = 5/nb_paintings
    listOfMethods = [createImageWithShapes] * nb_paintings + [createImageWithCategories] * nb_paintings

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

    if (len(object_image_list) == 0 or len (object_file_list) == 0):
        # List of candidate replacement objects
        for obj in MaskRCNNModel.class_names:
            object_file_list[obj] = [y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                                for ext in image_extensions] for y in x]
        for nested_list in object_file_list.values():
            for i in nested_list:
                object_image_list.append(cv2.imread(i, cv2.IMREAD_UNCHANGED))
    #object_image_list = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for nested_list in object_file_list.values() for i in nested_list] #iterate over all the values of the dict to get all images

    for painting_filename in painting_file_list[0:2]:
        print(painting_filename)
        painting_name = os.path.basename(painting_filename)
        painting = load_img(painting_filename)
        painting = img_to_array(painting)
        painting_width, painting_height = painting.shape[1], painting.shape[0]

        # Extract significant items from painting
        results = model.detect([painting], verbose=0, probability_criteria=0.7)
        r = results[0]

        cursor = 0
        j=0
        # Generate a number of altered forms of painting
        #NUMBER_OF_TRIES = nb_paintings
        for technic in listOfMethods:  # todo : change the cursor at each loop to have different images at the end
            ##
            # On doit enregistrer l'image à toutes les itérations car la fonction
            # putpixel ne fonctionne qu'après avoir sauvegardé les changements apportés
            # Ainsi on charge l'image temporaire, dans les cas où l'itération courante n'est pas la première
            ##
            print("Painting number : ", j)
            
            # Pick a random background image
            background_image_name = random.choice(background_file_list)
            background_image = Image.open(background_image_name)
            # Resize the background image with the size of painting.
            background_image = background_image.resize((painting_width, painting_height), Image.ANTIALIAS)
            background_image = background_image.convert("RGBA")
            
            background_image, realValue = technic(background_image, painting, r, cursor)
            #background_image, realValue = createImageWithShapes(background_image, painting, r, cursor)

            # Save background_image.
            file_saved = path_to_results + painting_name + str(j) + '.png'
            background_image = background_image.convert("RGB")
            background_image.save(file_saved)
            print("Real value obtained : ", realValue)
            cursor+=cursor_diff #to have different result for an image
            j+=1


if __name__ == "__main__":
    painting_dir = './Paintings/'
    background_dir = './BaseImages/paysages décors fonds'
    output_dir = './NewRésultats2/'

    create_case_study(painting_dir, './CroppedAndVectorizedImages/animaux/', background_dir, output_dir)
