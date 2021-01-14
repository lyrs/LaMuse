from glob import glob
import numpy as np
from matplotlib import pyplot as plt

from .MaskRCNNModel import MaskRCNNModel
from ..Mask_RCNN.mrcnn.visualize import display_instances
import argparse
import os
from PIL import Image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def generate_images(source_path: str, destination_path: str) -> None:
    """
    :param source_path: top level directory path in which to recursively select all images
    :param destination_path: destination path in which to generate image masks of detected objects
    :return:

    generate_images recursively iterates over all images in source_path and generates masks of detected objects
    in destination_path.

    If destination_path: does not exist it is created.
    """
    assert source_path != ""
    assert destination_path != ""

    image_extensions = ["jpg", "gif", "png", "tga"]
    image_list = [y for x in [glob(source_path + '/**/*.%s' % ext, recursive=True) for ext in image_extensions] for y in x]

    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    model = MaskRCNNModel().model
    compteur = 0

    # Iterate over all files in source_path
    # @Todo : verify that all files are effectively images, handle errors
    for filename in image_list:
        img = load_img(filename)
        original_image = load_img(filename)

        img = img_to_array(img)

        # Detecting known classes with RCNN model in image list [img]
        results = model.detect([img], verbose=0, probability_criteria=0.7)
        # Since image list contains only one element, the results array contains only one result
        r = results[0]

        # @Todo: make min_area run-time parameter
        min_area = 500

        # @Todo: consider taking into account a selection threshold on r['scores']

        ##
        # print('Trying to plot')
        # a_, ax = plt.subplots(1)
        # plt.ion()
        # plt.show()
        # display_instances(img, r['rois'], r['masks'], r['class_ids'], MaskRCNNModel.class_names, r['scores'], ax=ax)
        # plt.draw()
        # plt.pause(0.01)
        # print('end of plot instructions')
        ##

        if r['class_ids'].size > 0:

            for c in r['class_ids']:
                new_path = destination_path+'/'+MaskRCNNModel.class_names[c]
                if not os.path.exists(new_path):
                    os.mkdir(new_path)

            for obj_idx in range(len(r['rois'])):

                box_dimensions = r['rois'][obj_idx]
                box_area = abs((box_dimensions[1] - box_dimensions[3]) * (box_dimensions[0] - box_dimensions[2]))
                if box_area < min_area:
                    break

                masked_image = np.full(img.shape, (0, 0, 0), np.uint8)

                for c in range(3):
                    masked_image[:, :, c] = np.where(r['masks'][:, :, obj_idx] == 1,
                                                     img[:, :, c], 0)

                #segmented_image = Image.new("RGBA", masked_image)

                ##
                # following lines adapted from https://stackoverflow.com/questions/54703674/how-do-i-make-my-numpy-image-take-an-alpha-channel-value-in-python
                ##

                h, w = masked_image.shape[:2]
                # Adding an alpha layer to masked_image
                masked_image = np.dstack((masked_image, np.zeros((h, w), dtype=np.uint8) + 255))
                # Make mask of black pixels - mask is True where image is black
                mBlack = (masked_image[:, :, 0:3] == [0, 0, 0]).all(2)
                # Make all pixels matched by mask into transparent ones
                masked_image[mBlack] = (0, 0, 0, 0)

                segmented_image = Image.fromarray(masked_image)

                # print(filename, r['masks'][:, :, 0].shape[0], r['masks'][:, :, 0].shape[1], MaskRCNNModel.class_names[r['class_ids'][obj_idx]] + "/" + str(compteur) + "_" + str(obj_idx) + ".png")

                # Good dimensions crop([1], [0], [3], [2]) !
                segmented_image.crop(
                    (box_dimensions[1], box_dimensions[0], box_dimensions[3], box_dimensions[2])).save(
                     destination_path + "/" + MaskRCNNModel.class_names[r['class_ids'][obj_idx]] + "/" + str(compteur) + "_" +
                            str(obj_idx) + ".png")

            compteur += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="generate_segmented_pictures",
                                     description='Extract objects from the images provided in the input directory and '
                                                 'store the resulting segmented images in the output directory.')
    parser.add_argument("input_dir", metavar='in', type=str, nargs='?', help='input directory', default='./')
    parser.add_argument("output_dir", metavar='out', type=str, nargs='?',
                        help='output directory (defaults to input_dir if non specified)')

    args = parser.parse_args()
    print(args)
    generate_images(args.input_dir, args.output_dir if args.output_dir else args.input_dir)
