import numpy as np
import math

# Calculates the bounding box for the target.
class BboxInFieldOfVision:
    def __init__(self, target):
        # left, top, right, bottom
        self.target = target
        # field of view is 120 deg, 60 deg from middle = pi/3
        self.field_of_view = math.pi/3

    def set_target(self, target):
        self.target = target

    def get_bbox_angles(self,povs_dict, eyes_dict):
        povs = []
        keys = []
        eyes = []
        for k,v in povs_dict.items():
            povs.append(v)
            keys.append(k)
            eyes.append(eyes_dict[k])

        eyes = np.array(eyes)
        left_corner = np.array([self.target[0] - eyes[:, 0], self.target[1] - eyes[:, 1]]).transpose()
        right_corner = np.array([self.target[2] - eyes[:, 0], self.target[1] - eyes[:, 1]]).transpose()
        # povs = BboxInFieldOfVision.makeGaze2d(povs)

        angle_left = BboxInFieldOfVision.get_angle(left_corner, np.array(povs))
        angle_right = BboxInFieldOfVision.get_angle(right_corner, np.array(povs))

        angles = np.array([angle_left, angle_right]).transpose()
        angles_dict = {keys[i]:angles[i] for i in range(len(angles))}
        opposite = BboxInFieldOfVision.is_opposite(angles)
        opposite_dict = {keys[i]: opposite[i] for i in range(len(opposite))}
        return angles_dict, opposite_dict


    def get_bbox_Field_of_vision_angles(self, povs, eyes):
        angles, opposite = self.get_bbox_angles(povs,eyes)
        if np.any(opposite):
            print("The gaze is directed in ht opposite side of the target.")
        if np.any(np.logical_xor(np.array(abs(angles[:,0]-angles[:,1]) > math.pi/3*2), opposite)):
            print("?")
        same = np.invert(opposite)
        bbox_edges = np.empty_like(angles)

        bbox_edges[same,0] = np.amin(angles[same], axis=1) + self.field_of_view
        bbox_edges[same,1] = np.amax(angles[same], axis=1) - self.field_of_view

        bbox_edges[opposite,0] = np.amin(angles[opposite], axis=1) - self.field_of_view
        bbox_edges[opposite,1] = np.amax(angles[opposite], axis=1) + self.field_of_view

        too_small = BboxInFieldOfVision.angles_less_than_pi(bbox_edges[opposite,0])
        bbox_edges[opposite,0] = [x + 2*math.pi if too_small[i] else x for i,x in enumerate(bbox_edges[opposite,0])]

        too_large = BboxInFieldOfVision.angles_greater_than_pi(bbox_edges[opposite,1])
        bbox_edges[opposite,1] = [x - 2*math.pi if too_large[i] else x for i, x in enumerate(bbox_edges[opposite, 1])]

        return bbox_edges, opposite


    @staticmethod
    def is_opposite(angles):
        opposite_signs = np.logical_xor(angles[:,0] < 0, angles[:,1] < 0)
        opposite_side = abs(angles[:,0])+abs(angles[:,1]) > math.pi
        return  np.logical_and(opposite_signs,opposite_side)

    @staticmethod
    def angles_less_than_pi(angles):
        return angles < -math.pi

    @staticmethod
    def angles_greater_than_pi(angles):
        return angles > math.pi

    @staticmethod
    def get_angle(B, A, get_clockwise=True):
        def length_matrix(v):
            temp = v[:,0] ** 2 + v[:,1] ** 2
            return np.sqrt(temp)

        def dot_product(v, w):
            return v[:,0] * w[:,0] + v[:,1] * w[:,1]

        def determinant(v, w):
            return v[:,0] * w[:,1] - v[:,1] * w[:,0]

        def inner_angle(v, w):
            cosx = dot_product(v, w) / (length_matrix(v) * length_matrix(w))
            return np.arccos(cosx)  # in radians

        inner = inner_angle(A, B)
        if not get_clockwise:
            return inner;
        det = determinant(A, B)
        # if the det > 0 then A is immediately clockwise of B and the det <= 0 then B is clockwise of A
        # but in image x is opposite, so it is switched
        det = np.array(det < 0, dtype=bool)
        inner[det] = -inner[det]
        return inner


