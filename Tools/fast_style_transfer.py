import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL
from matplotlib import pyplot as plt


# tf.compat.v1.enable_eager_execution

##
# Code as described in https://www.tensorflow.org/tutorials/generative/style_transfer
##

def tensor_to_image(tensor):
    tensor = tf.cast(tensor * 255, dtype=tf.uint8)

    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img: str):
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
    print(new_shape)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def new_load_img(path_to_img: str, max_dim: int = 512):
    """
    :param path_to_img:
    :param max_dim:
    :return:
    """

    image = PIL.Image.open(path_to_img)
    shape = image.size
    long_dim = max(shape)
    scale = max_dim / long_dim

    image = image.resize((int(shape[0] * scale), int(shape[1] * scale)), PIL.Image.ANTIALIAS)
    # image.save(path_to_img + "_style.png")
    image = np.asarray(image)

    return image


def save_image(path_content: str, path_style: str, path_to_save: str) -> None:
    max_dim = 1024
    min_dim = 128  # A smaller image size will crash style transfer network

    # Load content and style images (see example in the attached colab).
    # https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2
    content_image = plt.imread(path_content)
    # style_image = plt.imread(path_content)

    style_image = new_load_img(path_style, max_dim)
    content_image_scaled = new_load_img(path_content, max_dim)

    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    content_image_scaled = content_image_scaled.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.

    # Load image stylization module.
    hub_module = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize image.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = tf.cast(outputs[0] * 255, dtype=tf.uint8).numpy()[0]

    outputs_scaled = hub_module(tf.constant(content_image_scaled), tf.constant(style_image))
    stylized_image_scaled = tf.cast(outputs_scaled[0] * 255, dtype=tf.uint8).numpy()[0]

    stylized_image = PIL.Image.fromarray(stylized_image)
    stylized_image.save(path_to_save)

    stylized_image_scaled = PIL.Image.fromarray(stylized_image_scaled)
    stylized_image_scaled.save(path_to_save + "_scaled.png")
