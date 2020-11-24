import matplotlib.patches as patches
import numpy as np
import math
from PIL import Image
from matplotlib.path import Path
from gaze360.gaze360 import Gaze360

class GazeToFieldOfVision():

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

        angle = GazeToFieldOfVision.angle(v)
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
    def toHeatmap(image, gazes, eyes, gazes_min, gazes_max):
        objects = []
        probability_images = []
        for i in range(len(gazes)):
            gaze = gazes[i]
            eye = eyes[i]
            min = gazes_min[i]
            max = gazes_max[i]
            gaze_vektor = Gaze360.makeGaze2d(gaze)

            rotate1 = Gaze360.makeGaze2d(min)#GazeToFieldOfVision.rotate(60 * var[i, 0], gaze_vektor)
            rotate2 = Gaze360.makeGaze2d(max)#GazeToFieldOfVision.rotate(-60 * var[i, 0], gaze_vektor)

            above = patches.Arrow(eye[0], eye[1], rotate1[0], rotate1[1], linewidth=1, edgecolor=(1, 0.5, 0), facecolor='none')
            real = patches.Arrow(eye[0], eye[1], gaze_vektor[0], gaze_vektor[1], linewidth=1, edgecolor=(1, 0, 0), facecolor='none')
            below = patches.Arrow(eye[0], eye[1], rotate2[0], rotate2[1], linewidth=1, edgecolor=(1, 0, 1), facecolor='none')
            map = GazeToFieldOfVision.get_probability_map(image.size, eye, gaze_vektor, rotate1, rotate2)

            # Poly
            #xy = [(eye[0], eye[1])]
            #long1 = GazeToFieldOfVision.getDiviation(image, eye, rotate1)
            #v1 = (eye[0] + rotate1[0] * long1, eye[1] + rotate1[1] * long1)
            #xy.append(v1)
            #xy += GazeToFieldOfVision.getAdditionalCorners(rotate1, image)
            #xy += GazeToFieldOfVision.getAdditionalCorners(gaze_vektor, image)
            #xy += GazeToFieldOfVision.getAdditionalCorners(rotate2, image)
            #long2 = GazeToFieldOfVision.getDiviation(image, eye, rotate2)
            #v2 = (eye[0] + rotate2[0] * long2, eye[1] + rotate2[1] * long2)
            #xy.append(v2)
            #xy.append((eye[0], eye[1]))
            #poly = patches.Polygon(xy, closed=True, color="b", alpha=0.1)

            # mask using poly
            #mask = GazeToFieldOfVision.get_mask(image.size, xy)
            #mask_real = GazeToFieldOfVision.get_mask(image.size, [(eye[0], eye[1]), (eye[0] + gaze[0] * long1, eye[1] + gaze[1] * long1)])
            #indices = np.where(mask_real)
            #distance = GazeToFieldOfVision.coordinates(np.array(mask_real).shape)
            #mask_real_3d = np.repeat(mask_real[:, :, np.newaxis], 2, axis=2)
            #distance_real = []
            #for i, x in enumerate(mask):
            #    distance_real = distance_real + [(i,j) for j, y in enumerate(x) if y]
            #mask = mask + mask_real
            objects += [above, real, below]#, poly]
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
        #return np.meshgrid(*(np.arange(s) for s in shape), indexing='ij')

    @staticmethod
    def get_probability(eye, unit_gaze, coordinate, a,b):
        # get vector to coordinate
        v = [coordinate[0]-eye[0], (coordinate[1]-eye[1])]

        # get angle to gaze
        unit_v = v / np.linalg.norm(v)
        angle_gaze = np.arccos(np.dot(unit_gaze, unit_v))

        # probability
        prob = int(a*angle_gaze + b)
        if prob < 0:
            prob = 0
        return [prob,prob,prob, prob]

    @staticmethod
    def get_probability_map(shape, eye, gaze, gaze_min, gaze_max):
        shape = GazeToFieldOfVision.swap(shape)
        shape.append(4)
        eye = GazeToFieldOfVision.swap(eye)
        gaze = GazeToFieldOfVision.swap(gaze)
        gaze_min = GazeToFieldOfVision.swap(gaze_min)
        gaze_max = GazeToFieldOfVision.swap(gaze_max)

        #coordinate_map = GazeToFieldOfVision.coordinates(shape);
        probability_map = np.zeros(shape, dtype=np.uint8)
        unit_gaze = gaze / np.linalg.norm(gaze)
        unit_gaze_min = gaze_min / np.linalg.norm(gaze_min)
        angle_gazemin = np.arccos(np.dot(unit_gaze_min, unit_gaze))
        unit_gaze_max = gaze_max / np.linalg.norm(gaze_max)
        angle_gazemax = np.arccos(np.dot(unit_gaze_max, unit_gaze))
        a,b = GazeToFieldOfVision.get_linear_function_params(0,(angle_gazemin+angle_gazemax)/2, 255, (255/100)*20)

        for i in range(0, shape[0], 2):
            for j in range(0, shape[1], 2):
                probability_map[i,j] = GazeToFieldOfVision.get_probability(eye, unit_gaze, (i,j), a,b)

        #probability_map = np.swapaxes(probability_map, 1, 0)
        return probability_map

    @staticmethod
    def swap(a):
        return [a[1], a[0]]

    @staticmethod
    def get_linear_function_params(x1,x2,y1,y2):
        a = (y2-y1)/(x2-x1)
        b = y1 - (a*x1)
        return a,b