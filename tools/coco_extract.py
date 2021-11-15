import argparse
import json
import numpy as np
import cv2
import os

from pycocotools.coco import COCO
from .MaskRCNNModel import MaskRCNNModel

'''
This module extracts series of overlay masks from the coco dataset
'''


def print_licenses(file: str) -> None:
    """
    Prints the available licenses of a COCO annotation file then exits

    :param file: JSON filename (supposedly compatible avec the COCO annotation format)
    :return: None
    """

    f = open(file, )
    data = json.load(f)

    for lic in data['licenses']:
        print(lic)

    f.close()
    exit()


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Constructs a new image with alpha-channel by extracting the pixels from the source image corresponding to the mask
    and setting other pixels to transparent.
    :param img: source image to apply mask on
    :param mask: mask image
    :return: new image with source image pixels on mask, and transparent otherwise
    """
    masked_image = np.full(img.shape, (0, 0, 0), np.uint8)

    for c in range(3):
        masked_image[:, :, c] = np.where(mask == 1, img[:, :, c], 0)

    ##
    # following lines adapted from https://stackoverflow.com/questions/54703674/how-do-i-make-my-numpy-image-take-an-alpha-channel-value-in-python
    ##
    h, w = masked_image.shape[:2]
    # Adding an alpha layer to masked_image
    masked_image = np.dstack((masked_image, np.zeros((h, w), dtype=np.uint8) + 255))
    # Make mask of black pixels - mask is True where image is black
    m_black = (masked_image[:, :, 0:3] == [0, 0, 0]).all(2)
    # Make all pixels matched by mask into transparent ones
    masked_image[m_black] = (0, 0, 0, 0)
    return masked_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="coco_extract",
                                     description='Extract objects from the images provided  by the COCO dataset and '
                                                 'store the resulting segmented images in the output directory.')
    parser.add_argument("--input_dir", "-in", metavar='in', type=str, nargs='?',
                        help='input directory (defaults to local directory if non specified)', default='./')
    parser.add_argument("--input_file", "-json", metavar='json', type=str,
                        help='input json file with COCO-compatible annotations')
    parser.add_argument("--output_dir", "-out", metavar='out', type=str, nargs='?',
                        help='output directory (defaults to input_dir if non specified)')
    parser.add_argument("--classes", "-cls", metavar='cls', type=str, nargs='+', default=MaskRCNNModel.class_names[1:],
                        help='classes to extract')
    parser.add_argument("--skip", "-skp", metavar='skp', type=str, nargs='+', default=[],
                        help='classes to skip')
    args = parser.parse_args()

    # Add coco dataset images folder path
    '''
    root_dir = './COCO'
    data_dir = f'{root_dir}/images/train2017'
    ann_file = f'{root_dir}/annotations/instances_train2017.json'
    seg_output_path = f'{root_dir}/seg'
    '''
    data_dir = args.input_dir
    seg_output_dir = args.input_dir
    if args.output_dir:
        seg_output_dir = args.output_dir
    ann_file = args.input_file

    skipClasses = args.skip

    coco = COCO(ann_file)
    # catIds = coco.getCatIds(catNms=['person']) #Add more categories ['person','dog']
    print(coco.info())
    exit(0)

    # Iterate over all wanted classes (available classes minus skipClasses)
    for className in [c for c in args.classes if c not in skipClasses]:
        catIds = coco.getCatIds(catNms=[className])
        imgIds = coco.getImgIds(catIds=catIds)

        nbImgs = len(imgIds)

        # Create directory to store extracted image patches of 'className' objects
        new_path = os.path.join(seg_output_dir, className)
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        # 'fullCount' = total number of extracted instances of class 'className'
        fullCount = 0
        # 'maxCount' = maximum allowed number of instances of class 'className'
        maxCount = 2000

        for imageIndex in range(nbImgs):
            img = coco.loadImgs(imgIds[imageIndex])[0]
            # Only keep images with appropriate licencing
            if not img['license'] in [4, 7, 8]:
                continue

            file_name = os.path.join(data_dir, img['file_name'])
            original_img = cv2.imread(file_name)

            if original_img is None:
                continue

            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
            anns = coco.loadAnns(annIds)

            minArea = 4096

            anns = [annotation for annotation in anns if annotation['area'] > minArea]

            for i in range(len(anns)):
                mask = coco.annToMask(anns[i])
                res = apply_mask(original_img, mask)
                x = int(anns[i]['bbox'][0])
                y = int(anns[i]['bbox'][1])
                w = int(anns[i]['bbox'][2])
                h = int(anns[i]['bbox'][3])

                crop_img = res[y:y + h, x:x + w]

                cv2.imwrite(os.path.join(new_path, f"{imageIndex}_{i}.png"), crop_img)
                fullCount += 1

            print(f"processed...{fullCount} - {imageIndex}/{nbImgs} {className}")
            if fullCount >= maxCount:
                break

    print("Done")
