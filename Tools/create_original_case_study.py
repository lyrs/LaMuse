import os

#os.chdir('./Mask_RCNN')
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import random
import cv2

import PIL
from PIL import Image
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from Tools.MaskRCNNModel import MaskRCNNModel

# draw an image with detected objects

def draw_image_with_boxes(filename, boxes_list):
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


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def create_case_study(path_paintings, path_objects_to_replace, path_images_fond, path_resultats):
    if not os.path.exists(path_resultats):
        os.mkdir(path_resultats)

    # Run the model on the painting.
    #@Todo find a way to invoke MaskRCNNModel only once
    model = MaskRCNNModel().model

    list_paintings = os.listdir(path_paintings)
    if '.DS_Store' in list_paintings:
        list_paintings.remove('.DS_Store')

    for painting_path in list_paintings:
        painting = load_img(path_paintings + painting_path)
        painting = img_to_array(painting)

        # make prediction

        results = model.detect([painting], verbose=0, probability_criteria=0.7)

        r = results[0]

        # Get the background image 

        ## Random choice
        list_background_images = os.listdir(path_images_fond)
        if '.DS_Store' in list_background_images:
            list_background_images.remove('.DS_Store')

        # Resize the background image with the size of the painting.

        painting_width, painting_height = painting.shape[1], painting.shape[0]

        # Parcours des objets reconnus et segmentés.

        NUMBER_OF_TRIES = 10

        ## List of Objects to replace.
        list_objects = os.listdir(path_objects_to_replace)
        if '.DS_Store' in list_objects:
            list_objects.remove('.DS_Store')
        for j in range(NUMBER_OF_TRIES):
            ## On change d'image de fond à tous les essais
            random_path = random.choice(list_background_images)

            background_image = Image.open(path_images_fond + '/' + random_path)
            background_image = background_image.resize((painting_width, painting_height), Image.ANTIALIAS)

            for i in range(r['class_ids'].size):
                ## On doit enregistrer l'image à toutes les itérations car la fonction 
                ## putpixel ne fonctionne qu'après avoir sauvegardé les changements apportés
                ## Ainsi on charge l'image temporaire, dans les cas où l'itération courante n'est pas la première 
                if i != 0:
                    background_image = Image.open(path_resultats + 'temp.png')

                ## Objet à coller
                object_to_replace = Image.open(path_objects_to_replace + random.choice(list_objects))

                ## Définition des dimensions et du placement du futur objet à coller

                boxes_img_to_replace = (r['rois'][i][1], r['rois'][i][0], r['rois'][i][3], r['rois'][i][2])
                width_img_to_replace = boxes_img_to_replace[2] - boxes_img_to_replace[0]
                height_img_to_replace = boxes_img_to_replace[3] - boxes_img_to_replace[1]

                object_to_replace = object_to_replace.resize((width_img_to_replace, height_img_to_replace),
                                                             Image.ANTIALIAS)

                ## Collage de l'image dans la peinture.

                for width in range(width_img_to_replace):
                    for height in range(height_img_to_replace):
                        value = object_to_replace.getpixel((width, height))
                        if value != (0, 0, 0, 0):
                            background_image.putpixel(
                                (boxes_img_to_replace[0] + width, boxes_img_to_replace[1] + height), value)
                # Save image.
                if i == r['class_ids'].size - 1:
                    file_saved = path_resultats + painting_path + str(j) + '.png'
                    background_image.save(file_saved)

                else:
                    background_image.save(path_resultats + 'temp.png')

                # 


if __name__ == "__main__":
    create_case_study('./Paintings/', './CroppedAndVectorizedImages/animaux/', './BaseImages/paysages décors fonds',
                      './NewRésultats2/')
