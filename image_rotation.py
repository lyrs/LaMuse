
import cv2
from glob import glob

import matplotlib.pyplot as plt

from .tools.MaskRCNNModel import MaskRCNNModel
from .tools.compare_images import applyOrientation, best_image, blackAndWhitePNG


def getcontour(image):
    img_gray = cv2.cvtColor(blackAndWhitePNG(image), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contour, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # contour of the target

    print("Number of Contours found = " + str(len(contour)))
    for i in (0,len(contour)-1):
        print(contour[i].shape)

    # Draw all contours
    # -1 signifies drawing all contours
    cv2.drawContours(image, contour, -1, (0, 255, 0), 3)

    #cv2.imshow('Contours', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return contour


if __name__ == "__main__":
    list_images_helico = ['186_0d', '186_45d', '186_45g', '186_90d', '186_90g', '186_180d']
    list_images_tigre = ['2_1', '2_1_90g', '2_1_20g', '2_1_45g', '2_1_89g', '2_1_90d(270g)', '2_1_120g', '2_1_180']

    path_objects_to_replace = "./LaMuse/BaseImages_objets"
    image_extensions = ["jpg", "gif", "png", "tga"]

    # All possible images
    object_file_list = []
    for obj in MaskRCNNModel.class_names:
        object_file_list.extend([y for x in [glob(path_objects_to_replace + '/%s/*.%s' % (obj, ext))
                                               for ext in image_extensions] for y in x])
    
    object_image_list = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in object_file_list]
    for i in range (12) :
        print("--------------", str(i), "--------------")
        target_image = cv2.imread('./LaMuse/rotate_image_testim/peinture/'+ str(i) +'.png', cv2.IMREAD_UNCHANGED) 
        #print(target_image)
        img = best_image(target_image, object_image_list, 0)[0]
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(str(i))
        axs[0].imshow(target_image)
        axs[0].set_title("image de base")
        axs[1].imshow(img)
        axs[1].set_title("image modèle")
        plt.show()

    default_image = cv2.imread('./LaMuse/rotate_image_testim/tigre/2_1.png', cv2.IMREAD_UNCHANGED)
    cd = getcontour(default_image)
    for img in list_images_tigre :
        print("--------------", img, "--------------")
        rotated_image = cv2.imread('./LaMuse/rotate_image_testim/tigre/'+ img +'.png', cv2.IMREAD_UNCHANGED)        
        finale_image = applyOrientation(cd, default_image, getcontour(rotated_image), rotated_image)

        fig, axs = plt.subplots(1, 3)
        fig.suptitle(img)
        axs[0].imshow(default_image)
        axs[0].set_title("image de base")
        axs[1].imshow(rotated_image)
        axs[1].set_title("image modèle")
        axs[2].imshow(finale_image)
        axs[2].set_title("image finale")
        plt.show()

    default_image = cv2.imread('./LaMuse/rotate_image_testim/helicoptere/186_0d.png', cv2.IMREAD_UNCHANGED)
    cd = getcontour(default_image)
    for img in list_images_helico :
        print("--------------", img, "--------------")
        rotated_image = cv2.imread('./LaMuse/rotate_image_testim/helicoptere/'+ img +'.png', cv2.IMREAD_UNCHANGED)        
        finale_image = applyOrientation(cd, default_image, getcontour(rotated_image), rotated_image)

        fig, axs = plt.subplots(1, 3)
        fig.suptitle(img)
        axs[0].imshow(default_image)
        axs[0].set_title("image de base")
        axs[1].imshow(rotated_image)
        axs[1].set_title("image modèle")
        axs[2].imshow(finale_image)
        axs[2].set_title("image finale")
        plt.show()