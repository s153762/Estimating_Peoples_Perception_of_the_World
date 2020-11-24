from PIL import Image
import cv2
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import face_recognition
import time
import numpy as np

from detecting_attended_targets.detecting_attended_targets import DetectingAttendedTargets
from gaze360.gaze360 import Gaze360
from gaze_field_of_vision import GazeToFieldOfVision

class EstimatingIndividualsPerspective:
    def __init__(self):
        print("Starting estimating individual's perspectives")

    def main(self):
        detectingAttendedTargets = DetectingAttendedTargets('detecting_attended_targets/model_weights/model_demo.pt',)
        gaze360 = Gaze360('gaze360/gaze360_model.pth.tar',)

        axs = self.setup_plot(2)
        plt.ion()

        vc = cv2.VideoCapture('../../TrainingSet/test.mp4');#TwoPersons.m4v');#
        image_raw, image = self.grab_frame(vc)
        ims = []
        ims.append(axs[0].imshow(image));
        ims.append(axs[1].imshow(image));

        while vc.isOpened():
            for i in range(4):
                image_raw, image = self.grab_frame(vc)
                if image_raw is None:
                    break;

            if image_raw is None:
                vc.release();
                cv2.destroyAllWindows()
                continue;

            face_locations, face_landmarks = self.get_face_locations(image_raw, True, True)
            black_image = Image.new('RGBA', image.size, (0, 0, 0, 255))
            if len(face_locations) > 0:
                heatmap, blended = self.plot_detecting_attended_targets(detectingAttendedTargets, image, face_locations)
                ims[1].set_data(blended)
                ims[0].set_data(image)

                gazes, eyes, min, max = self.plot_gaze360(gaze360, image, face_locations, face_landmarks)
                polygons, prob_image = GazeToFieldOfVision.toHeatmap(image, gazes, eyes, min, max)
                #heatmap_array = np.array(Image.alpha_composite(black_image, heatmap))
                #mask_array = np.array(mask)
                #result = cv2.bitwise_and(heatmap_array, mask_array)
                #ims[1].set_data(mask_array)
                heatmapGaze = image.convert("RGBA")
                for image in prob_image:
                    heatmapGaze = Image.alpha_composite(heatmapGaze, image.convert("RGBA"))
                ims[0].set_data(heatmapGaze)

                for ply in polygons:
                    axs[0].add_patch(ply)

                #for arrow in arrows+heads:
                #    axs[0].add_patch(arrow)

                plt.pause(0.0001)
                #for arrow in arrows+heads:
                #    arrow.remove()
                for ply in polygons:
                    ply.remove()
            else:
                ims[0].set_data(image)
                ims[1].set_data(image)#black_image)
                plt.pause(0.0001)


        plt.ioff()
        plt.show()

    def plot_detecting_attended_targets(self, detectingAttendedTargets, image, face_locations):
        heatmap = detectingAttendedTargets.getHeatmap(image, face_locations, True)
        return heatmap, Image.alpha_composite(image.convert("RGBA"), heatmap)#Image.alpha_composite(Image.new('RGBA', image.size, (0, 0, 0, 0)), heatmap), Image.alpha_composite(image.convert("RGBA"), heatmap) #Image.blend(image.convert("RGBA"), heatmap, alpha=.5)


    def plot_gaze360(self, gaze360, image, face_locations, face_landmarks):
        return gaze360.getArrows(image, face_locations, face_landmarks, True, False)
        #return arrows, var


    def grab_frame(self, vc):
        ret, image_raw = vc.read()
        if not ret or image_raw is None:
            return None, None
        frame = Image.fromarray(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))
        return image_raw, frame.convert('RGB')

    def get_face_locations(self, image, get_face_landmarks = True, printTime = False):
        starttime = None
        if printTime:
            starttime = time.time()
        face_locations = face_recognition.face_locations(image)
        if get_face_landmarks:
            face_landmarks = face_recognition.face_landmarks(image)
        if printTime:
            print("Time taken for face recognition: ", time.time() - starttime)
        if get_face_landmarks:
            return face_locations, face_landmarks
        return face_locations, None

    def setup_plot(self, number_of_axis, titles = ["Gaze360","DAVTV"]):
        fig = plt.figure(figsize=(10, 5))  # 15,10))
        fig.canvas.manager.window.move(0, 0);
        axs = []
        for axis in range(1,number_of_axis+1):
            axs.append(plt.subplot(1, number_of_axis, axis))
            axs[axis-1].set_axis_off()
            axs[axis - 1].set_title(titles[axis - 1])

        return axs


if __name__ == "__main__":
    EstimatingIndividualsPerspective().main()
