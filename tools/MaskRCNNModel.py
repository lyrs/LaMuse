import os

from ..Mask_RCNN.mrcnn.config import Config
from ..Mask_RCNN.mrcnn import model as modellib


class MaskRCNNModel:
    # Singleton pattern from : https://stackoverflow.com/questions/12305142/issue-with-singleton-python-call-two-times-init
    _instance = None
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(MaskRCNNModel, cls).__new__(cls)
            cls._instance.__initialized = False

        return cls._instance

    class MaskRCNNConfig(Config):
        # give the configuration a recognizable name
        NAME = "MaskRCNN_inference"

        # set the number of GPUs to use along with the number of images
        # per GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # number of classes (we would normally add +1 for the background
        # but the background class is *already* included in the class
        # names)
        NUM_CLASSES = 1 + 80

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    # @TODO properly include stuff using pkg_ressources
    def __init__(self):
        if(self.__initialized): 
            return
        self.__initialized = True
        self.config = MaskRCNNModel.MaskRCNNConfig()

        print("loading  weights for Mask R-CNN model…")
        self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir="./")

        path = os.path.dirname(__file__) + "/../mask_rcnn_coco.h5"
        self.model.load_weights(path, by_name=True)        
