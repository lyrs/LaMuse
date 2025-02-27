#  Copyright (c) 2022. Bart Lamiroy (Bart.Lamiroy@univ-reims.fr) and subsequent contributors
#  as per git commit history. All rights reserved.
#
#  La Muse, Leveraging Artificial Intelligence for Sparking Inspiration
#  https://hal.archives-ouvertes.fr/hal-03470467/
#
#  This code is licenced under the GNU LESSER GENERAL PUBLIC LICENSE
#  Version 3, 29 June 2007
#

import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
# import PIL
import cv2
import os
from matplotlib import pyplot as plt


# tf.compat.v1.enable_eager_execution

##
# Code as described in https://www.tensorflow.org/tutorials/generative/style_transfer
##

def tensor_to_image(tensor):
    """
    :param tensor:
    :return:
    """
    tensor = tf.cast(tensor * 255, dtype=tf.uint8)

    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    return tensor.numpy()
    #return PIL.Image.fromarray(tensor)


def load_img(path_to_img: str):
    """
    :param path_to_img:
    :return:
    """
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # shape = img.shape[:-1]

    # long_dim = max(shape)
    long_dim = tf.math.reduce_max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def new_load_img(path_to_img: str, max_dim: int = None):
    """
    :param path_to_img:
    :param max_dim:
    :return:
    """

    # image = PIL.Image.open(path_to_img)
    # image = image.convert("RGB")

    image = cv2.imread(path_to_img, cv2.IMREAD_COLOR)

    if max_dim:
        # shape = image.size
        shape = image.shape
        long_dim = max(shape)
        scale = max_dim / long_dim

        # image = image.resize((int(shape[0] * scale), int(shape[1] * scale)), PIL.Image.ANTIALIAS)
        image = cv2.resize(image, (int(shape[0] * scale), int(shape[1] * scale)), interpolation=cv2.INTER_LANCZOS4)

    # Make sure to remove transparency layer
    # image = np.asarray(image)[:, :, :3]
    return image


def apply_style_transfer(path_content: str, path_style: str, path_to_save: str, scale_image: bool = True) -> np.array:
    """
    :param path_content:
    :param path_style:
    :param path_to_save:
    :param scale_image:
    :return:
    """
    max_dim = 512
    min_dim = 128  # A smaller image size will crash style transfer network

    # Load content and style images (see example in the attached colab).
    # https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2
    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.
    # style_image = plt.imread(path_style)
    style_image = new_load_img(path_style, max_dim)

    if scale_image:
        content_image = new_load_img(path_content, max_dim)
    else:
        content_image = new_load_img(path_content, None)



    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    # Load image stylization module.
    os.environ['TFHUB_CACHE_DIR'] = './tf_cache'  # Any folder that you can access
    hub_module = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize image.
    # @TODO Investigate and check image formats correctly
    # @Bug Sometimes, for some kind of images,
    #  tensorflow aborts with the following error : tensorflow.python.framework.errors_impl.InvalidArgumentError:
    #  input depth must be evenly divisible by filter depth: 4 vs 3
    #
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = tf.cast(outputs[0] * 255, dtype=tf.uint8).numpy()[0]
    stylized_image = np.asarray(stylized_image)

    # stylized_image = PIL.Image.fromarray(stylized_image)
    # stylized_image = stylized_image.convert("RGB")
    # stylized_image.save(path_to_save)
    cv2.imwrite(path_to_save, stylized_image)

    return stylized_image

    # Stylize image (scaled).
    # outputs_scaled = hub_module(tf.constant(content_image_scaled), tf.constant(style_image))
    # stylized_image_scaled = tf.cast(outputs_scaled[0] * 255, dtype=tf.uint8).numpy()[0]
    # stylized_image_scaled = PIL.Image.fromarray(stylized_image_scaled)
    # stylized_image_scaled.save(path_to_save + "_scaled.png")
