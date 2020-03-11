import os

os.chdir('../Mask_RCNN')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2

import PIL
from PIL import Image
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class myMaskRCNNConfig(Config):
 # give the configuration a recognizable name
 NAME = "MaskRCNN_inference"
 
 # set the number of GPUs to use along with the number of images
 # per GPU
 GPU_COUNT = 1
 IMAGES_PER_GPU = 1
 
 # number of classes (we would normally add +1 for the background
 # but the background class is *already* included in the class
 # names)
 NUM_CLASSES = 1+80

config = myMaskRCNNConfig()

print("loading  weights for Mask R-CNN model…")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="./")

model.load_weights("mask_rcnn_coco.h5", by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
 'bus', 'train', 'truck', 'boat', 'traffic light',
 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
 'kite', 'baseball bat', 'baseball glove', 'skateboard',
 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
 'teddy bear', 'hair drier', 'toothbrush']

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

# Run the model on the painting.

list_paintings = os.listdir('../Paintings/')
if '.DS_Store' in list_paintings:
    list_paintings.remove('.DS_Store')

for painting_path in list_paintings:
    painting = load_img('../Paintings/'+'800px-Piero_della_Francesca_042.jpg')
    painting = img_to_array(painting)

    # make prediction

    results = model.detect([painting], verbose=0, probability_criteria= 0.7)

    r = results[0]

    # Get the background image 

    ## Random choice
    list_background_images = os.listdir('../BaseImages/paysages décors fonds/')
    if '.DS_Store' in list_background_images:
        list_background_images.remove('.DS_Store')

    # Resize the background image with the size of the painting.

    painting_width, painting_height = painting.shape[1], painting.shape[0]


    # Parcours des objets reconnus et segmentés.

    NUMBER_OF_TRIES = 10 

    ## List of Objects to replace.
    list_objects = os.listdir('../CropedAndVectorizedImages./personnages/')
    if '.DS_Store' in list_objects:
        list_objects.remove('.DS_Store')
    for j in range(NUMBER_OF_TRIES):
        ## On change d'image de fond à tous les essais
        random_path = random.choice(list_background_images)
        
        
        background_image = Image.open('../CropedAndVectorizedImages./personnages'+random.choice(list_objects))
        background_image = background_image.resize((painting_width, painting_height), Image.ANTIALIAS)

        for i in range(r['class_ids'].size):
            ## On doit enregistrer l'image à toutes les itérations car la fonction 
            ## putpixel ne fonctionne qu'après avoir sauvegardé les changements apportés
            ## Ainsi on charge l'image temporaire, dans les cas où l'itération courante n'est pas la première 
            if i != 0:
                background_image = Image.open('../Resultats/temp.png')

            ## Objet à coller
            object_to_replace = Image.open('../CropedAndVectorizedImages./personnages/' + random.choice(list_objects))

            ## Définition des dimensions et du placement du futur objet à coller

            boxes_img_to_replace = (r['rois'][i][1], r['rois'][i][0], r['rois'][i][3], r['rois'][i][2])
            width_img_to_replace = boxes_img_to_replace[2] - boxes_img_to_replace[0]
            height_img_to_replace = boxes_img_to_replace[3] - boxes_img_to_replace[1]

            object_to_replace = object_to_replace.resize((width_img_to_replace,height_img_to_replace), Image.ANTIALIAS)

            ## Collage de l'image dans la peinture.

            for width in range(width_img_to_replace):
                for height in range(height_img_to_replace):
                    value =  object_to_replace.getpixel((width, height))
                    if value != (0,0,0,0):
                        background_image.putpixel((boxes_img_to_replace[0]+width, boxes_img_to_replace[1]+height),value)
            # Save image.
            if i == r['class_ids'].size -1 :
                background_image.save('../Resultats/' + painting_path +str(j)+ 'WithStyleTransfer'+'.png')
            else:
                background_image.save('../Resultats/temp.png')

