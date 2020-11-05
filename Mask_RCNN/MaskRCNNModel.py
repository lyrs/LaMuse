from mrcnn.config import Config
from mrcnn import model as modellib


class MaskRCNNModel:
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

    def __init__(self):
        self.config = MaskRCNNModel.MaskRCNNConfig()

        print("loading  weights for Mask R-CNN model…")
        self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir="./")
        self.model.load_weights("mask_rcnn_coco.h5", by_name=True)
