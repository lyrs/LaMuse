from glob import glob

from Tools.MaskRCNNModel import MaskRCNNModel
import argparse
import os
from PIL import Image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


##
#   generate_images iterates over all images in source_path and generates masks of detected objects in destination_path
##
def generate_images(source_path: str, destination_path: str) -> None:
    assert source_path != ""
    assert destination_path != ""

    image_extensions = ["jpg", "gif", "png", "tga"]
    image_list = [y for x in [glob(source_path + '/*.%s' % ext) for ext in image_extensions] for y in x]

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

        if r['class_ids'].size > 0:

            for obj_idx in range(len(r['rois'])):

                box_dimensions = r['rois'][obj_idx]
                if abs((box_dimensions[1] - box_dimensions[3]) * (box_dimensions[0] - box_dimensions[2])) < min_area:
                    break

                vectorized_image = Image.new("RGBA", original_image.size, 0)

                for height in range(r['masks'][:, :, 0].shape[0]):
                    for width in range(r['masks'][:, :, 0].shape[1]):
                        if r['masks'][height, width, 0]:
                            # @Todo: rather than using box area, use pixel threshold to filter image size
                            vectorized_image.putpixel((width, height), original_image.getpixel((width, height)))

                # Good dimensions crop([1], [0] [3] [2]) !
                vectorized_image.crop(
                    (box_dimensions[1], box_dimensions[0], box_dimensions[3], box_dimensions[2])).save(
                    destination_path + "/" + MaskRCNNModel.class_names[r['class_ids'][obj_idx]] + str(compteur) + "_" +
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
