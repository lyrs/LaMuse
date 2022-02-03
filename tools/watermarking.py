#  Copyright (c) 2022. Bart Lamiroy (Bart.Lamiroy@univ-reims.fr) and subsequent contributors
#  as per git commit history. All rights reserved.
#
#  La Muse, Leveraging Artificial Intelligence for Sparking Inspiration
#  https://hal.archives-ouvertes.fr/hal-03470467/
#
#  This code is licenced under the GNU LESSER GENERAL PUBLIC LICENSE
#  Version 3, 29 June 2007
#

# Inspired by https://www.analyticsvidhya.com/blog/2021/08/how-to-add-watermark-on-images-using-opencv-in-python/

import numpy as np
import cv2


def add_watermark(img: np.ndarray, path_to_watermark: str = "./Watermark.png") -> np.ndarray:
    imgHeight, imgWidth = img.shape[:2]
    new_dim = (imgWidth, imgHeight)

    overlay = cv2.imread(path_to_watermark, cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.size == 0:
        raise OSError("Trying to read {} returns empty image".format(path_to_watermark))

    overlay = cv2.resize(overlay, new_dim, interpolation=cv2.INTER_AREA)

    # Extract the RGB channels
    srcRGB = img[..., :3]
    dstRGB = overlay[..., :3]

    # Extract the alpha channels and normalise to range 0..1
    srcA = img[..., 3] / 255.0
    dstA = overlay[..., 3] / 255.0

    # Work out resultant alpha channel
    outA = srcA + dstA * (1 - srcA)

    # Work out resultant RGB
    # outRGB = (srcRGB * srcA[..., np.newaxis] + dstRGB * srcA[..., np.newaxis] * (1 - dstA[..., np.newaxis])) / outA[
    #    ..., np.newaxis]
    new_srcRGB = srcRGB * (srcA-dstA)[..., np.newaxis]
    new_dstRGB = (dstRGB * dstA[..., np.newaxis] + srcRGB * srcA[..., np.newaxis]) / (dstA + srcA)[..., np.newaxis]
    outRGB = new_srcRGB + new_dstRGB * (1-(srcA-dstA))[..., np.newaxis]

    # Merge RGB and alpha (scaled back up to 0..255) back into single image
    outRGBA = np.dstack((outRGB, outA * 255)).astype(np.uint8)

    return outRGBA

