import matplotlib.patches as patches
import numpy as np
import math
from gaze360.Gaze360 import Gaze360

class CombinationMethod():

    @staticmethod
    def rotate(degree, gaze):
        theta = np.radians(degree)
        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        return r.dot(np.array(gaze))

    @staticmethod
    def getDiviation(image, eye, rotate):
        width, height = image.size
        maxWidth = width - eye[0]
        maxheight = height - eye[1]
        if rotate[0] < 0:
            maxWidth = eye[0]
        if rotate[1] < 0:
            maxheight = eye[1]
        return min(maxWidth / abs(rotate[0]), maxheight / abs(rotate[1]))

    @staticmethod
    def getAdditionalCorners(v, image):
        width, height = image.size
        xy = []
        #print("SHOW:", v[0], v_new[0])
        #if v[0] < 5 and v_new[0] > 5:
            #xy.append((width,0))

        angle = CombinationMethod.angle(v)
        if angle >= 0 and angle < math.pi / 2:
            xy.append((width, 0))
        elif angle >= math.pi / 2 and angle < math.pi:
            xy.append((0, 0))
        elif angle >= math.pi and angle < math.pi * 1.5:
            xy.append((0, height))
        elif angle >= math.pi*1.5 and angle <= math.pi * 2:
            xy.append((width, height))
        return xy

    @staticmethod
    def angle(B):
        def length(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2)

        def dot_product(v, w):
            return v[0] * w[0] + v[1] * w[1]

        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]

        def inner_angle(v, w):
            cosx = dot_product(v, w) / (length(v) * length(w))
            return math.acos(cosx)  # in radians

        A = [1,0]
        inner = inner_angle(A, B)
        det = determinant(A, B)
        if det < 0:  # this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else:  # if the det > 0 then A is immediately clockwise of B
            return math.pi*2 - inner

    @staticmethod
    def toHeatmap(image, gazes, eyes):
        objects = []
        for i in range(len(gazes)):
            gaze = gazes[i]
            eye = eyes[i]

            gaze_vektor = Gaze360.makeGaze2d(gaze)
            xy = [(eye[0], eye[1])]


            rotate1 = CombinationMethod.rotate(60, gaze_vektor)
            rotate2 = CombinationMethod.rotate(-60, gaze_vektor)
            long1 = CombinationMethod.getDiviation(image, eye, rotate1)
            long2 = CombinationMethod.getDiviation(image, eye, rotate2)
            v1 = (eye[0] + rotate1[0] * long1, eye[1] + rotate1[1] * long1)
            v2 = (eye[0] + rotate2[0] * long2, eye[1] + rotate2[1] * long2)

            above = patches.Arrow(eye[0], eye[1], rotate1[0], rotate1[1], linewidth=1, edgecolor=(1, 0.5, 0),
                                  facecolor='none')
            real = patches.Arrow(eye[0], eye[1], gaze_vektor[0], gaze_vektor[1], linewidth=1, edgecolor=(1, 0, 0),
                                 facecolor='none')
            below = patches.Arrow(eye[0], eye[1], rotate2[0], rotate2[1], linewidth=1, edgecolor=(1, 0, 1),
                                  facecolor='none')

            xy.append(v1)
            xy += CombinationMethod.getAdditionalCorners(rotate1, image)
            xy += CombinationMethod.getAdditionalCorners(gaze_vektor, image)
            xy += CombinationMethod.getAdditionalCorners(rotate2, image)
            xy.append(v2)
            xy.append((eye[0], eye[1]))
            poly = patches.Polygon(xy, closed=True, color="b", alpha=0.1)
            poly.
            objects += [above, real, below, poly]
        return objects
