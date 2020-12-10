import matplotlib.patches as patches
import numpy as np
import math
from PIL import Image
from matplotlib.path import Path
from gaze360.gaze360 import Gaze360

class GazeToFieldOfVision():

    def __init__(self, target):
        # left, top, right, bottom
        self.target = target

    def set_target(self, target):
        # left, top, right, bottom
        self.target = target

    def get_probabilities(self, prob_images):
        probs = np.array([np.asarray(image) for image in prob_images])
        targets = probs[:,self.target[0]:self.target[2], self.target[1]:self.target[3],3]
        mean = np.mean(targets,axis=(1,2))
        return mean/255*100

    @staticmethod
    def toHeatmap(image, gazes, eyes, gazes_min, gazes_max):
        objects = []
        probability_images = []
        for i in range(len(gazes)):
            gaze = gazes[i]
            eye = eyes[i]
            min = gazes_min[i]
            max = gazes_max[i]
            gaze_vektor = Gaze360.makeGaze2d(gaze)

            rotate1 = Gaze360.makeGaze2d(min)
            rotate2 = Gaze360.makeGaze2d(max)

            above = patches.Arrow(eye[0], eye[1], rotate1[0], rotate1[1], linewidth=1, edgecolor=(1, 0.5, 0), facecolor='none')
            real = patches.Arrow(eye[0], eye[1], gaze_vektor[0], gaze_vektor[1], linewidth=1, edgecolor=(1, 0, 0), facecolor='none')
            below = patches.Arrow(eye[0], eye[1], rotate2[0], rotate2[1], linewidth=1, edgecolor=(1, 0, 1), facecolor='none')
            map = GazeToFieldOfVision.get_probability_map(image.size, eye, gaze_vektor, rotate1, rotate2)

            objects += [above, real, below]
            probability_images += [Image.fromarray(map)]

        return objects, probability_images

    def getArray(self, width, height, poly_verts):
        nx, ny = width, height
        codes = [Path.MOVETO]
        for i in range(len(poly_verts[1:-1])):
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)

        path = Path(poly_verts, codes)
        return path

    @staticmethod
    def get_mask(shape, verts):
        """Return image mask given by mask creator"""
        w, h = shape
        y, x = np.mgrid[:h, :w]
        points = np.transpose((x.ravel(), y.ravel()))
        path = Path(verts)
        mask = path.contains_points(points)
        mask = mask.reshape(h, w)
        result = np.zeros((h,w,4))
        for i in range(4):
            result[:,:,i][mask] = True
        return mask

    @staticmethod
    def coordinates(shape):
        return np.indices(shape).transpose((1,2,0))


    @staticmethod
    def get_probability_map(shape, eye, gaze, gaze_min, gaze_max):
        shape = GazeToFieldOfVision.swap(shape)

        eye = GazeToFieldOfVision.swap(eye)
        gaze = GazeToFieldOfVision.swap(gaze)
        gaze_min = GazeToFieldOfVision.swap(gaze_min)
        gaze_max = GazeToFieldOfVision.swap(gaze_max)

        # broadcasting solution
        coordinate_map = GazeToFieldOfVision.coordinates(shape);
        coordinate_map[:, :, 0] = coordinate_map[:, :, 0] - eye[0] # get vector to coordinate
        coordinate_map[:, :, 1] = coordinate_map[:, :, 1] - eye[1] # get vector to coordinate
        map = GazeToFieldOfVision.angle_matrix(gaze, coordinate_map)

        angle_gaze_min = GazeToFieldOfVision.get_angle(gaze_min, gaze) #np.arccos(np.dot(unit_gaze_min, unit_gaze))
        angle_gaze_max = GazeToFieldOfVision.get_angle(gaze_max, gaze) #np.arccos(np.dot(unit_gaze_max, unit_gaze))
        a,b = GazeToFieldOfVision.get_linear_function_params(0,max(angle_gaze_min,angle_gaze_max), 255, (255/100)*20)
        map = map*a+b;
        map[map < 0] = 0
        map[map > 255] = 255
        probability_map = np.empty(shape, dtype=np.uint8)
        probability_map.fill(255)
        probability_map = np.array([probability_map]*4, dtype=np.uint8);
        probability_map[3] = np.array(map).astype(int)
        return probability_map.transpose((1,2,0))


    @staticmethod
    def get_angle(B, A, get_clockwise = False):
        def length(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2)

        def dot_product(v, w):
            return v[0] * w[0] + v[1] * w[1]

        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]

        def inner_angle(v, w):
            cosx = dot_product(v, w) / (length(v) * length(w))
            return math.acos(cosx)  # in radians

        inner = inner_angle(A, B)
        det = determinant(A, B)
        if det < 0 or not get_clockwise:  # this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else:  # if the det > 0 then A is immediately clockwise of B
            return - inner #math.pi*2 - inner

    @staticmethod
    def swap(a):
        return [a[1], a[0]]

    @staticmethod
    def get_linear_function_params(x1,x2,y1,y2):
        a = (y2-y1)/(x2-x1)
        b = y1 - (a*x1)
        return a,b

    @staticmethod
    def angle_matrix(A, B_matrix, get_clockwise=False):
        def length(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2)

        def length_matrix(v):
            temp = v[:,:,0] ** 2 + v[:,:,1] ** 2
            return np.sqrt(temp)

        def dot_product(v, w):
            return v[0] * w[:,:,0] + v[1] * w[:,:,1]

        def determinant(v, w):
            return v[0] * w[:,:,1] - v[1] * w[:,:,0]

        def inner_angle(v, w):
            cosx = dot_product(v, w) / (length(v) * length_matrix(w))
            return np.arccos(cosx)  # in radians

        inner = inner_angle(A, B_matrix)
        if not get_clockwise:
            return inner;
        det = determinant(A, B_matrix)
        det = np.array(det > 0, dtype=bool) # if the det > 0 then A is immediately clockwise of B and the det <= 0 then B is clockwise of A
        inner[det] = -inner[det]
        return inner