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
import os
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
    NUM_CLASSES = 1 + 80


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

def draw_image_with_boxes(filename : str, boxes_list) -> None:
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

if __name__ == "__main__":
    img = load_img('../../_108857562_mediaitem108857561.jpg')
    copy_img = load_img('../../_108857562_mediaitem108857561.jpg')
    img = img_to_array(img)

    # make prediction
    results = model.detect([img], verbose=0, probability_criteria=0.88)

    r = results[0]

    milkmaid_img = load_img('../../Vermeer_-_The_Milkmaid.jpg')
    milkmaid_img = img_to_array(milkmaid_img)

    results_milkmaid = model.detect([milkmaid_img], verbose=0, probability_criteria=0.88)

    r_milkmaid = results_milkmaid[0]

    # show photo with bounding boxes, masks, class labels and scores
    # display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize= (11,7))


    # Construct vectorized image :
    original_image = Image.open('../../_108857562_mediaitem108857561.jpg')
    vectorized_image = Image.new("RGBA", original_image.size, 0)

    ## the masks returned back is in reverse order : (height, width) instead of (width,height).

    # height
    for height in range(r['masks'][:, :, 0].shape[0]):
        # width
        for width in range(r['masks'][:, :, 0].shape[1]):

            if r['masks'][height, width, 0]:
                vectorized_image.putpixel((width, height), original_image.getpixel((width, height)))

    # vectorized_image.show()

    ## Paste the vectorized image on the milkmaid

    milkmaid = Image.open('../../Vermeer_-_The_Milkmaid.jpg')
    """
    milkmaid.paste(vectorized_image)
    milkmaid.show()
    """

    ## Directly transfer the vectorized image on a destination image.

    destination_image = Image.open('../../Vermeer_-_The_Milkmaid.jpg')

    # height
    for height in range(r['masks'][:, :, 0].shape[0]):
        # width
        for width in range(r['masks'][:, :, 0].shape[1]):

            width_to_add = r_milkmaid['rois'][0][0]
            height_to_add = r_milkmaid['rois'][0][1]

            if r['masks'][height, width, 0]:
                destination_image.putpixel((width + width_to_add, height + height_to_add),
                                       original_image.getpixel((width, height)))

    destination_image.show()

    destination_image.save('./la_laitière_avec_oiseau_incrusté.jpg')

    # Handle boxes, places where the masks will take place.

    # Parcours des masques.
    for i in range(r['class_ids'].size):
        # Gestion des boxes
        # choix du masque
        # Resize ?
        print("ok")
