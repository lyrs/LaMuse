
import cv2


import matplotlib.pyplot as plt


from .tools.compare_images import applyOrientation, blackAndWhitePNG


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

    default_image = cv2.imread('./LaMuse/rotate_image_testim/tigre/2_1.png', cv2.IMREAD_UNCHANGED)
    cd = getcontour(default_image)
    for img in list_images_tigre :
        print("--------------", img, "--------------")
        rotated_image = cv2.imread('./LaMuse/rotate_image_testim/tigre/'+ img +'.png', cv2.IMREAD_UNCHANGED)        
        finale_image = applyOrientation(cd, rotated_image, getcontour(rotated_image), default_image)

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
        finale_image = applyOrientation(cd, rotated_image, getcontour(rotated_image), default_image)

        fig, axs = plt.subplots(1, 3)
        fig.suptitle(img)
        axs[0].imshow(default_image)
        axs[0].set_title("image de base")
        axs[1].imshow(rotated_image)
        axs[1].set_title("image modèle")
        axs[2].imshow(finale_image)
        axs[2].set_title("image finale")
        plt.show()