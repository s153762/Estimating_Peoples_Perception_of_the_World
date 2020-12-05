import time
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.patches as patches
import math
#import face_recognition

from gaze360.model import GazeLSTM

class Gaze360:
    model = None;
    transforms_normalize = None
    input_images = None
    image_count = 0

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
        output[:, 1] = torch.cos(azimuthal_angle) * torch.sin(polar_angle)
        output[:, 0] = torch.sin(azimuthal_angle)
        output[:, 2] = -torch.cos(azimuthal_angle)*torch.cos(polar_angle)
        return output

    def getEyes(self, head_box):
        # TODO: Improve eye location
        # eyes[0], eyes[1] = eyes[0] / float(image.shape[1]), eyes[1] / float(image.shape[0])
        # 2*eyes[0]-1, -2*eyes[1]+1
        eyes = [(head_box[0] + head_box[2]) / 2.0, (0.65 * head_box[1] + 0.35 * head_box[3])]
        return np.asarray(eyes).astype(float)

    def get_eyes(self, face_landmark):
        right = np.mean(face_landmark["right_eye"], axis=0)
        left = np.mean(face_landmark["left_eye"], axis=0)
        eyes = left+(right-left)/2
        return np.asarray(eyes).astype(float)


    def makeArrows(self, head_box, gaze, min, max):
        eyes = self.get_eyes(head_box)
        gaze = Gaze360.makeGaze2d(gaze)
        return patches.Arrow(eyes[0], eyes[1], gaze[0], gaze[1], linewidth=2, edgecolor=(1, 0, 0),
                                    facecolor='none')

    @staticmethod
    def makeGaze2d(gaze):
        gaze = [-gaze[0],gaze[1],gaze[2]]
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


    def getHeadbox(self, image, face_locations, face_landmarks):
        count = 0
        input_image = torch.zeros(len(face_locations), 7, 3, 224, 224) #self.input_images[:len(face_locations), :,:,:,:]
        head_boxs = []
        #eyes = []

        for face_location in face_locations:
            top, right, bottom, left = face_location
            head_boxs.append([left, top, right, bottom])
            head = image.crop((head_boxs[count]))  # head crop
            input_image[count, self.image_count, :, :, :] = self.transforms_normalize(head)
            self.input_images[count, self.image_count, :, :, :] = input_image[count, self.image_count, :, :, :]
            count += 1
        #for face_landmark in face_landmarks:
        #    eyes.append(self.get_eyes(face_landmark))

        self.image_count = (self.image_count + 1) % 7
        return input_image, head_boxs, face_landmarks

    def getGaze(self, input_image, amount):
        # forward pass
        input = input_image.view(amount, 7, 3, 224, 224)
        output_gaze, var = self.model(input)  # .cuda())
        #var = var.detach().numpy()
        gazes = self.spherical2cartesial(output_gaze).detach().numpy()
        gazes_min = self.spherical2cartesial(output_gaze-var).detach().numpy()
        gazes_max = self.spherical2cartesial(output_gaze+var).detach().numpy()
        print("Var: ", var)
        print("Gazes: ", gazes, gazes_min, gazes_max)
        return gazes, gazes_min, gazes_max

    def getArrows(self, image, face_locations, face_landmarks, printTime, returnArrows):
        arrows = []
        heads = []
        starttime = None
        if printTime:
            starttime = time.time()

        input_image, head_boxs, eyes = self.getHeadbox(image, face_locations, face_landmarks)
        gazes, min, max = self.getGaze(input_image, len(face_locations))
        if not returnArrows:
            return gazes, eyes, min, max

        for i in range(len(gazes)):
            head_box = head_boxs[i]
            arrows.append(self.makeArrows(head_box, gazes[i], min, max))
            heads.append(self.makeHeadboxes(head_box))

        if printTime:
            print("Time taken to estimate gaze369: ", time.time() - starttime)
        return arrows, heads

    @staticmethod
    def getEyeLocalization(image):
        eye_localizations = []
        face_landmarks_list = face_recognition.face_landmarks(image)
        for face_landmarks in face_landmarks_list:
            eye_localizations.append(Gaze360.midpointTwoCoordinates(face_landmarks['left_eye'], face_landmarks['right_eye']))
        return eye_localizations

    @staticmethod
    def midpointTwoCoordinates(p1, p2):
        return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2]
