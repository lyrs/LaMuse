# Inspired from https://www.analyticsvidhya.com/blog/2021/08/how-to-add-watermark-on-images-using-opencv-in-python/

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


def add_watermark_old(img: np.ndarray, path_to_watermark: str = "./Watermark.png") -> np.ndarray:

    watermark = cv2.imread(path_to_watermark, cv2.IMREAD_UNCHANGED)

    # Rescaling the watermark image
    w_img = int(img.shape[1])
    h_img = int(img.shape[0])

    # @Bug this will break with greylevel images !!
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        ##
        # b_channel, g_channel, r_channel = cv2.split(img)
        # alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype)   # creating a dummy alpha channel image.
        # img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        ##

    print("Input channels {}".format(img.shape[2]));
    print("WM channels {}".format(watermark.shape[2]));

    new_dim = (w_img, h_img)
    resized_wm = cv2.resize(watermark, new_dim, interpolation=cv2.INTER_AREA)

    center_y = int(h_img / 2)
    center_x = int(w_img / 2)
    h_wm, w_wm, _ = resized_wm.shape
    top_y = center_y - int(h_wm / 2)
    left_x = center_x - int(w_wm / 2)
    bottom_y = top_y + h_wm
    right_x = left_x + w_wm

    roi = img[top_y:bottom_y, left_x:right_x]
    result = cv2.addWeighted(roi, 1, resized_wm, 0.3, 0)
    img[top_y:bottom_y, left_x:right_x] = result

    return img
