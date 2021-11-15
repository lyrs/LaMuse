import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from ..tools.compare_images import best_image
from ..tools.MaskRCNNModel import MaskRCNNModel
from glob import glob

from ..setup import *


def draw_comparison_graph(max_ratio: int) -> None:
    # @TODO : make a graph with time to compare efficiency
    x = []
    y = []
    ratio_correct = []

    path_objects_to_replace = default_substitute_folder

    # All possible images
    object_file_list = []
    for obj in MaskRCNNModel.class_names:
        object_file_list.extend([y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                             for ext in image_extensions] for y in x])
    object_image_list = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in object_file_list]
    shuffle(object_image_list)
    empty = np.empty((0, 0, 4))
    # image_list_unchanged = object_image_list.copy()
    for i in range(2, max_ratio):

        res_add = 0
        print("Ratio : ", i)
        k = 0
        ratio = 0

        for j in range(len(object_file_list) - 1, 0, -200):
            target_image = object_image_list[j]
            object_image_list[j] = empty
            res, ratio = best_image(target_image, object_image_list, 0.0, i)[1:]
            res_add = res_add + res
            object_image_list[j] = target_image
            k += 1
        if k > 0:
            x.append(i)
            y.append(res_add / k)
            ratio_correct.append(ratio)
        # object_image_list = image_list_unchanged.copy()
    plt.figure(1)
    plt.title("Bizarritude moyenne en fonction du ratio chosi")
    plt.plot(x, y)

    plt.figure(2)
    plt.title("Nombre d'images correspondant au ratio")
    plt.plot(x, ratio_correct)
    plt.show()


draw_comparison_graph(20)
