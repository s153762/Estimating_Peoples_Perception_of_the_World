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
        self.input_images = torch.zeros(3, 7, 3, 224, 224)

        # Model
        self.model = GazeLSTM()
        # self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)  # .cuda()
        checkpoint = torch.load(model_weights, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.image_people_count = 0
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
        count = 0
        input_image = self.input_images[:len(face_locations), :,:,:,:] #torch.zeros(len(face_locations), 7, 3, 224, 224) #
        head_boxs = []

        for face_location in face_locations:
            top, right, bottom, left = face_location
            head_boxs.append([left, top, right, bottom])
            head = image.crop((head_boxs[count]))  # head crop
            if self.image_people_count <= count:
                for i in range(7):
                    input_image[count, i, :, :, :] = self.transforms_normalize(head)
                self.image_people_count += 1
                self.input_images[count] = input_image[count]
            else:
                self.input_images[count] = Gaze360.push_to_tensor(self.input_images[count],self.transforms_normalize(head))
            #self.input_images[count, self.image_count, :, :, :] = input_image[count, self.image_count, :, :, :]
            count += 1

        #self.image_count = (self.image_count + 1) % 7
        return self.input_images[:len(face_locations)], head_boxs

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

        input_image, head_boxs = self.get_headbox(image, face_locations)
        gazes, gazes_10, gazes_90 = self.get_gaze(input_image, len(face_locations))
        if get2D:
            gazes = np.array([self.gaze_2d(gaze) for gaze in gazes])
            gazes_10 = np.array([self.gaze_2d(gaze) for gaze in gazes_10])
            gazes_90 = np.array([self.gaze_2d(gaze) for gaze in gazes_90])

        self.gaze360_time += time.time() - starttime
        if printTime:
            print("Time taken to estimate gaze360: ", time.time() - starttime)
        return gazes, gazes_10, gazes_90

    @staticmethod
    def midpoint_two_coordinates(p1, p2):
        return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
