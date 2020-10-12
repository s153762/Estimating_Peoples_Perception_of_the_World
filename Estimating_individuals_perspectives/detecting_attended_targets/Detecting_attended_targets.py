import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
import cv2

from detecting_attended_targets.model import ModelSpatial
from detecting_attended_targets.utils import imutils, evaluation
from detecting_attended_targets.config import *

class DetectingAttendedTargets:
    model_weights = None;
    transforms_normalize = None;
    model = None;

    def __init__(self, model_weights):
        print('Starting detecting attended targets')
        self.model_weights = model_weights;

        # set up data transformation
        self.transforms_normalize = self._get_transform()

        # Model
        self.model = ModelSpatial()
        model_dict = self.model.state_dict()
        checkpoint = torch.load(self.model_weights, map_location=lambda storage, loc: storage)
        checkpoint = checkpoint['model']
        model_dict.update(checkpoint)
        self.model.load_state_dict(model_dict)
        # self.model.cuda()
        self.model.train(False)

    def _get_transform(self):
        transform_list = []
        transform_list.append(transforms.Resize((input_resolution, input_resolution)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)

    def getHeatmap(self, image, face_locations, printTime):
        with torch.no_grad():
            width, height = image.size

            first = True
            heatmap = Image.new('RGBA', (width, height), (0, 0, 0, 225));
            starttime = None;

            if printTime:
                starttime = time.time()

            for face_location in face_locations:
                heatmap_new = self.getSingleHeatmap(image, face_location, width, height)
                if first:
                    first = False;
                else:
                    heatmap = Image.blend(heatmap, heatmap_new, alpha=.5)
        if printTime:
            print("Time taken to estimate attended targets: ", time.time()-starttime)
        return ImageOps.invert(heatmap.convert('RGB'));

    def getSingleHeatmap(self, image, face_location, width, height):
        top, right, bottom, left = face_location
        head_box = [left - 30, top - 45, right + 30, bottom + 45]
        head = image.crop((head_box))  # head crop

        head = self.transforms_normalize(head)  # transform inputs
        frame = self.transforms_normalize(image)

        head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3],
                                                    width, height, resolution=input_resolution).unsqueeze(0)
        head = head.unsqueeze(0)  # .cuda()
        frame = frame.unsqueeze(0)  # .cuda()
        head_channel = head_channel.unsqueeze(0)  # .cuda()

        # forward pass
        raw_hm, _, inout = self.model(frame, head_channel, head)

        # heatmap modulation
        raw_hm = raw_hm.cpu().detach().numpy() * 255
        raw_hm = raw_hm.squeeze()
        inout = inout.cpu().detach().numpy()
        inout = 1 / (1 + np.exp(-inout))
        inout = (1 - inout) * 255
        norm_map = np.array(Image.fromarray(raw_hm).resize((width, height))) - inout

        return Image.fromarray(norm_map).convert('RGBA')