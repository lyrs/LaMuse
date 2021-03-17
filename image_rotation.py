
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

    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return contour


if __name__ == "__main__":
    default_image = cv2.imread('./LaMuse/rotate_image_testim/helicoptere/186_0d.png', cv2.IMREAD_UNCHANGED)
    rotated_image = cv2.imread('./LaMuse/rotate_image_testim/helicoptere/186_45d.png', cv2.IMREAD_UNCHANGED)

    cd = getcontour(default_image)
    finale_image = applyOrientation(cd, getcontour(rotated_image), default_image)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(default_image)
    axs[0].set_title("image de base")
    axs[1].imshow(rotated_image)
    axs[1].set_title("image modèle")
    axs[2].imshow(finale_image)
    axs[2].set_title("image finale")
    plt.show()
