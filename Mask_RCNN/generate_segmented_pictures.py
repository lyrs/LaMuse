from mrcnn.config import Config
from mrcnn import model as modellib
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

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


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


def generate_images(source_path, destination_path):
    assert source_path != ""
    assert destination_path != ""

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    compteur = 0

    for filename in os.listdir(source_path):
        img = load_img(source_path + '/' + filename)
        original_image = load_img(source_path + '/' + filename)

        img = img_to_array(img)

        results = model.detect([img], verbose=0, probability_criteria=0.7)
        r = results[0]

        vectorized_image = Image.new("RGBA", original_image.size, 0)

        if r['class_ids'].size > 0:
            for height in range(r['masks'][:, :, 0].shape[0]):
                for width in range(r['masks'][:, :, 0].shape[1]):
                    if r['masks'][height, width, 0]:
                        vectorized_image.putpixel((width, height), original_image.getpixel((width, height)))
            boxes_dimensions = r['rois'][0]
            # Good dimensions crop([1], [0] [3] [2]) !
            vectorized_image.crop(
                (boxes_dimensions[1], boxes_dimensions[0], boxes_dimensions[3], boxes_dimensions[2])).save(
                destination_path + "/" + class_names[r['class_ids'][0]] + str(compteur) + ".png")
            compteur += 1


if __name__ == "__main__":
    generate_images("../BaseImages/objets", "../NouvelEssai")
