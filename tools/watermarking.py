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
    return add_diagonal_watermark(img, path_to_watermark)


def add_diagonal_watermark(img: np.ndarray, path_to_watermark: str = "./Watermark.png") -> np.ndarray:
    img_height, img_width = img.shape[:2]
    new_dim = (img_width, img_height)

    overlay = cv2.imread(path_to_watermark, cv2.IMREAD_UNCHANGED)
    if overlay is None or overlay.size == 0:
        raise OSError("Trying to read {} returns empty image".format(path_to_watermark))

    overlay = cv2.resize(overlay, new_dim, interpolation=cv2.INTER_AREA)

    # Extract the RGB channels
    src_rgb = img[..., :3]
    dst_rgb = overlay[..., :3]

    # Extract the alpha channels and normalise to range 0..1
    src_alpha = img[..., 3] / 255.0
    dst_alpha = overlay[..., 3] / 255.0

    # Work out resultant alpha channel
    out_alpha = src_alpha + dst_alpha * (1 - src_alpha)

    # Work out resultant RGB
    # out_rgb = (src_rgb * src_alpha[..., np.newaxis] + dst_rgb * src_alpha[..., np.newaxis] * (1 - dstA[..., np.newaxis])) / out_alpha[
    #    ..., np.newaxis]
    new_src_rgb = src_rgb * (src_alpha - dst_alpha)[..., np.newaxis]
    new_dst_rgb = (dst_rgb * dst_alpha[..., np.newaxis] + src_rgb * src_alpha[..., np.newaxis]) / (dst_alpha + src_alpha)[..., np.newaxis]
    out_rgb = new_src_rgb + new_dst_rgb * (1 - (src_alpha - dst_alpha))[..., np.newaxis]

    # Merge RGB and alpha (scaled back up to 0..255) back into single image
    out_rgba = np.dstack((out_rgb, out_alpha * 255)).astype(np.uint8)

    return out_rgba
