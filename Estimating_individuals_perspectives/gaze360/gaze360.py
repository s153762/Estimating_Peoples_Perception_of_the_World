import time
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.patches as patches
import math
from collections import deque
from torch import Tensor
#import face_recognition

from gaze360.model import GazeLSTM

class Gaze360:
    model = None;
    transforms_normalize = None
    input_images = None


    def __init__(self, model_weights):
        print('Starting Gaze360')
        WIDTH, HEIGHT = 960, 720
        self.transforms_normalize = self._get_transform()
        self.input_images = {}

        # Model
        self.model = GazeLSTM()
        # self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)  # .cuda()
        checkpoint = torch.load(model_weights, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.gaze360_time = 0

    def _get_transform(self):
        transform_list = []
        transform_list.append(transforms.Resize((224, 224)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform_list)

    def spherical2cartesial(self, x):
        output = torch.zeros(x.size(0), 3)
        polar_angle = x[:, 0]
        azimuthal_angle = x[:, 1]
        output[:, 1] = torch.sin(azimuthal_angle)
        output[:, 0] = torch.cos(azimuthal_angle) * torch.sin(polar_angle)
        output[:, 2] = -torch.cos(azimuthal_angle)*torch.cos(polar_angle)
        return output

    #@staticmethod # First implementation
    #def makeGaze2d(gaze):
    #    gaze[:, 0] = -gaze[:, 0]
    #    n = [0, 0, 1]
    #    def length(v):
    #        return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    #    def dot_product(v, w):
    #        return v[0] * w[:, 0] + v[1] * w[:, 1] + v[2] * w[:, 2]
    #    def times(v, n):
    #        return [v[0] * n, v[1] * n, v[2] * n]
    #    def minus(v, w):
    #        return np.array([v[:, 0] - w[0], v[:, 1] - w[1], v[:, 2] - w[2]])
    #    proj = times(n, (dot_product(n, gaze) / (length(n) ** 2)))
    #    test = minus(gaze, proj)
    #    return test[:-1].transpose() * 100

    @staticmethod
    def gaze_2d(gaze):
        gaze = [-gaze[0],-gaze[1],-gaze[2]]#[-gaze[1],gaze[0],gaze[2]]
        n = [0,0,1]
        def length(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

        def dot_product(v, w):
            return v[0] * w[0] + v[1] * w[1] + v[2] * w[2]

        def times(v,n):
            return [v[0]*n,v[1]*n,v[2]*n]

        def minus(v,w):
            return  [v[0]-w[0],v[1]-w[1],v[2]-w[2]]

        proj = times(n,(dot_product(n,gaze) / (length(n) ** 2)))
        test = minus(gaze,proj)
        return times(test,100)[:-1]

    def makeHeadboxes(self, head_box):
        return patches.Rectangle((head_box[0], head_box[1]), head_box[2] - head_box[0], head_box[3] - head_box[1],
                              linewidth=1, edgecolor=(0, 1, 1), facecolor='none')

    @staticmethod
    def push_to_tensor(tensor, x):
        return torch.cat((tensor[1:], x.unsqueeze(0)))

    def get_headbox(self, image, face_locations):
        input_images = torch.zeros(len(face_locations), 7, 3, 224, 224)
        head_boxs = {}
        keys = []
        face_count = 0
        for key, face_location in face_locations.items():
            top, right, bottom, left = face_location
            head_boxs[key] = [left, top, right, bottom]
            head = image.crop((head_boxs[key]))  # head crop
            if key not in self.input_images.keys():
                input = torch.zeros(7, 3, 224, 224)
                for i in range(7):
                    input[i] = self.transforms_normalize(head)
                self.input_images[key] = input
            else:
                self.input_images[key] = Gaze360.push_to_tensor(self.input_images[key],
                                                                self.transforms_normalize(head))
            input_images[face_count, :, :, :, :] = self.input_images[key]
            keys.append(key)
            face_count+=1

        return input_images, head_boxs.values(), keys

    def get_gaze(self, input_image, amount):
        # forward pass
        input = input_image.view(amount, 7, 3, 224, 224)
        output_gaze, var = self.model(input)  # .cuda())
        #var = var.detach().numpy()
        gazes = self.spherical2cartesial(output_gaze).detach().numpy()
        gazes_10 = self.spherical2cartesial(output_gaze-var).detach().numpy()
        gazes_90 = self.spherical2cartesial(output_gaze+var).detach().numpy()
        return gazes, gazes_10, gazes_90

    def get_gaze_direction(self, image, face_locations, printTime, get2D = True):
        starttime = time.time()

        input_image, head_boxs, keys = self.get_headbox(image, face_locations)
        gazes, gazes_10, gazes_90 = self.get_gaze(input_image, len(face_locations))
        if get2D:
            gazes = {keys[i]:self.gaze_2d(gazes[i]) for i in range(len(gazes))}
            gazes_10 = {keys[i]:self.gaze_2d(gazes_10[i]) for i in range(len(gazes))}
            gazes_90 = {keys[i]:self.gaze_2d(gazes_90[i]) for i in range(len(gazes))}

        self.gaze360_time += time.time() - starttime
        if printTime:
            print("Time taken to estimate gaze360: ", time.time() - starttime)
        return gazes, gazes_10, gazes_90

    @staticmethod
    def midpoint_two_coordinates(p1, p2):
        return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
