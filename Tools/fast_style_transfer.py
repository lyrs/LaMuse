import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import PIL


def tensor_to_image(tensor):
    tensor = tensor * 255
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
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def save_image(path_content: str, path_style: str, path_to_save: str) -> None:
    content = load_img(path_content)
    style = load_img(path_style)

    hub_module = tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    content_modified = tf.constant(content)
    style_modified = tf.constant(style)

    stylized_image = hub_module(content_modified, style_modified)[0]

    img = tensor_to_image(stylized_image)

    img.save(path_to_save)
