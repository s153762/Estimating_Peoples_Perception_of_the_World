# https://gist.github.com/tibaes/35b9dbd7cbf81a98955067aa318290e7#file-video
# https://medium.com/@kostal91/displaying-real-time-webcam-stream-in-ipython-at-relatively-high-framerate-8e67428ac522
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=ZyAvNCJMmvFF

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class Detectron2Keypoints:

    def __init__(self):
        self.cfg = get_cfg()
        if (not torch.cuda.is_available()):
            self.cfg.MODEL.DEVICE = "cpu"

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        # Predictor
        self.predictor = DefaultPredictor(self.cfg)

    def get_keypoints(self, frame):
        outputs = self.predictor(frame)
        v = Visualizer(frame, MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        instances = outputs["instances"].to("cpu")
        bbox = []
        keypoints = []
        pred_boxes = instances._fields["pred_boxes"].tensor.numpy()
        pred_keypoints = instances._fields["pred_keypoints"].numpy()
        for i in range(len(instances._fields["scores"])):
            if instances._fields["scores"][i]>0.99 and pred_keypoints[i].shape[0] == 17:
                bbox.append(pred_boxes[i])
                keypoints.append(pred_keypoints[i])

        keypoints = np.array(keypoints)
        bbox = np.array(bbox)
        eyes = Detectron2Keypoints.keypoints_to_eyes(keypoints) if len(keypoints) > 0 else []
        head_bbox = Detectron2Keypoints.bbox_keypoints_head_bbox(bbox, keypoints) if len(keypoints) > 0 else []
        return out.get_image(), head_bbox, eyes

    @staticmethod
    def keypoints_to_eyes(keypoints):
        eyes_keypoints = keypoints[:,:5, :2]
        eyes = eyes_keypoints.sum(axis=1)
        eyes = eyes / 5
        return eyes

    @staticmethod
    def bbox_keypoints_head_bbox(bbox, keypoints):
        # top, right, bottom, left
        sholders = keypoints[:,5:7,:2]
        buttom = sholders[:, :, 1].sum(axis=1) / 2
        top = bbox[:, 1]
        diff = (buttom - top) / 7
        left_ears = np.min(keypoints[:, :5, 0], axis = 1)
        left = (bbox[:, 0] + left_ears)/2
        right_ears = np.max(keypoints[:, :5, 0], axis = 1)
        right = (bbox[:, 2] +right_ears)/2

        head = np.array([top-diff, right, buttom-diff, left]).astype(int)
        head = head.T
        return head
