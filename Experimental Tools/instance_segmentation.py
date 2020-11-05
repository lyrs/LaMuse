"""
 This code looks like it is an tentative experiment for replacing one detected object in a given painting by one
 coming from a mystery_image regardless of type and shape
"""
import random

from PIL import Image
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from glob import glob

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from Tools.MaskRCNNModel import MaskRCNNModel


def draw_image_with_boxes(filename: str, boxes_list: list) -> None:
    """ draw an image with detected objects """

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


if __name__ == "__main__":

    background_dir = './BaseImages/paysages décors fonds/'
    image_extensions = ["jpg", "gif", "png", "tga"]

    background_image_list = [y for x in [glob(background_dir + '/*.%s' % ext) for ext in image_extensions] for y in x]

    mystery_image_path = random.choice(background_image_list)

    img = load_img(mystery_image_path)
    img = img_to_array(img)

    # Extract objects from mystery image
    model = MaskRCNNModel().model
    results = model.detect([img], verbose=0, probability_criteria=0.88)

    r = results[0]

    milkmaid_img_path = './Paintings/Vermeer_-_The_Milkmaid.jpg'
    milkmaid_img = load_img(milkmaid_img_path)
    milkmaid_img = img_to_array(milkmaid_img)

    results_milkmaid = model.detect([milkmaid_img], verbose=0, probability_criteria=0.88)

    r_milkmaid = results_milkmaid[0]

    # show photo with bounding boxes, masks, class labels and scores
    # display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize= (11,7))

    # Construct vectorized image :
    original_image = Image.open(mystery_image_path)
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

    milkmaid = Image.open(milkmaid_img_path)
    """
    milkmaid.paste(vectorized_image)
    milkmaid.show()
    """

    ## Directly transfer the vectorized image on a destination image.

    destination_image = Image.open(milkmaid_img_path)

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

    final_output_image = './la_laitière_avec_oiseau_incrusté.jpg'
    destination_image.save(final_output_image)

    # Handle boxes, places where the masks will take place.

    # Parcours des masques.
    for i in range(r['class_ids'].size):
        # Gestion des boxes
        # choix du masque
        # Resize ?
        print("ok")
