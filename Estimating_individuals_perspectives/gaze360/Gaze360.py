import time
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.patches as patches

from gaze360.model import GazeLSTM

class Gaze360:
    model = None;
    transforms_normalize = None

    def __init__(self, model_weights):
        print('Starting Gaze360')
        WIDTH, HEIGHT = 960, 720
        self.transforms_normalize = self._get_transform()


        # Model
        self.model = GazeLSTM()
        # model.cuda()
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
        output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
        output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
        output[:, 1] = torch.sin(x[:, 1])
        return output

    def getArrows(self, image, face_locations, printTime):
        arrows = []
        heads = []
        starttime = None
        if printTime:
            starttime = time.time()

        count = 0
        input_image = torch.zeros(2, 7, 3, 224, 224)
        head_boxs = []
        for face_location in face_locations:
            top, right, bottom, left = face_location
            head_boxs.append([left, top, right, bottom])
            head = image.crop((head_boxs[count]))  # head crop
            print(head_boxs)
            if count < 7:
                input_image[count, 0, :, :, :] = self.transforms_normalize(head)
                print(count)
            count+=1

        # forward pass
        output_gaze, var = self.model(input_image.view(2, 7, 3, 224, 224))  # .cuda())
        print("var", var)
        gazes = self.spherical2cartesial(output_gaze).detach().numpy()
        print("Gaze: ", gazes)

        for i in range(len(gazes)):
            gaze = gazes[i]
            print("Gaze: ", gaze)

            # TODO: Improve eye location
            eyes = [(head_boxs[i][0] + head_boxs[i][2]) / 2.0, (0.65 * head_boxs[i][1] + 0.35 * head_boxs[i][3])]
            eyes = np.asarray(eyes).astype(float)
            # eyes[0], eyes[1] = eyes[0] / float(image.shape[1]), eyes[1] / float(image.shape[0])

            heads.append(patches.Rectangle((head_boxs[i][0], head_boxs[i][1]), head_boxs[i][2] - head_boxs[i][0], head_boxs[i][3] - head_boxs[i][1],
                                     linewidth=1, edgecolor=(0, 1, 1), facecolor='none'))
            # 2*eyes[0]-1, -2*eyes[1]+1

            arrows.append(patches.Arrow(eyes[0], eyes[1], -gaze[2] * 200, gaze[1] * 200, linewidth=2, edgecolor=(1, 0, 0),
                                  facecolor='none'))

        if printTime:
            print("Time taken to estimate gaze369: ", time.time() - starttime)
        return arrows, heads
