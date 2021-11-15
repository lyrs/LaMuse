import os

version_number = '0.2.0'

image_extensions = ["jpg", "gif", "png", "tga", "jpeg"]


segmentation_suffix = "_objets"
default_base_folder = os.path.dirname(__file__)

print(f"Default execution directory = {default_base_folder}")

# @Todo: Currently configuration data is packed with the software and stored in the /bin or /lib
#   directory after installation/deployment. This should be changed to a more convenient location
default_image_folder = f'{default_base_folder}/BaseImages'
default_background_folder = f'{default_base_folder}/Backgrounds'
default_substitute_folder = default_image_folder + segmentation_suffix
default_painting_folder = f'{default_base_folder}/Paintings'
default_interpretation_folder = './Interpretations'
default_watermark_file = f'{default_base_folder}/Watermark.png'

mask_rcnn_config_file = f'{default_base_folder}/mask_rcnn_coco.h5'


