from glob import glob
from random import randint
import matplotlib.pyplot as plt
import cv2

from matplotlib.pyplot import imshow
import os

from .MaskRCNNModel import MaskRCNNModel


# from .compare_images import transparancy_mask_to_BW


def barycentre(image):  # entrée : Image, Sortie : abssice et ordonnée du barycentre
    # gray_image = transparancy_mask_to_BW(image)/255
    # print(image.shape)
    N = image.shape[0]
    M = image.shape[1]
    bar_abs = 0
    bar_ord = 0
    cptB = 0
    mask = image[:, :, 3] == 0  # 1 if transparent else 0
    for i in range(N):
        for j in range(M):
            if not mask[i, j]:  # | image[i, j][1] != 0 | image[i, j][2] != 0 | image[i, j][3] != 0):
                cptB = cptB + 1
                bar_abs = bar_abs + i
                bar_ord = bar_ord + j
    # print(N, M)
    # print(bar_abs, bar_ord)
    if (cptB == 0):
        # imshow(image)
        # plt.show()
        return -1, -1
    bar_ord = bar_ord / cptB
    bar_abs = bar_abs / cptB
    return int(bar_abs), int(bar_ord)


def visu_barycentre(image, bar_abs, bar_ord):
    for i in range(4):
        image[int(bar_abs) - i, int(bar_ord) - i][0] = 255
        image[int(bar_abs) - i, int(bar_ord) - i][1] = 255
        image[int(bar_abs) - i, int(bar_ord) - i][2] = 255
        image[int(bar_abs) - i, int(bar_ord) - i][3] = 255
        image[int(bar_abs) + i, int(bar_ord) + i][0] = 255
        image[int(bar_abs) + i, int(bar_ord) + i][1] = 255
        image[int(bar_abs) + i, int(bar_ord) + i][2] = 255
        image[int(bar_abs) + i, int(bar_ord) + i][3] = 255
    return image


if __name__ == "__main__":
    segmentation_suffix = "_objets"

    default_image_folder = './LaMuse/BaseImages'
    default_background_folder = None
    default_painting_folder = './LaMuse/Paintings'
    default_interpretation_folder = './Interpretations'

    mask_rcnn_config_file = os.path.dirname(__file__) + '/mask_rcnn_coco.h5'

    path_objects_to_replace = "./LaMuse/BaseImages_objets"
    image_extensions = ["jpg", "gif", "png", "tga"]

    object_file_list = []
    for obj in MaskRCNNModel.class_names:
        object_file_list.extend([y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                             for ext in image_extensions] for y in x])
    r = randint(0, len(object_file_list) - 1)
    target_file = object_file_list[r]
    target_image = cv2.imread(target_file, cv2.IMREAD_UNCHANGED)
    a, b = barycentre(target_image)
    print(a, b)
    image_bar = visu_barycentre(target_image, a, b)
    imshow(image_bar)
    plt.show()
