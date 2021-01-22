from PIL import Image
import cv2
import matplotlib
matplotlib.use('macosx')#.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import face_recognition
import time
import numpy as np

from detecting_attended_targets.detecting_attended_targets import DetectingAttendedTargets
from gaze360.gaze360 import Gaze360
from gaze_field_of_vision import GazeToFieldOfVision
from detectron.detecron2_keypoint import Detectron2Keypoints
from bbox_in_field_of_vision import BboxInFieldOfVision

class EstimatingIndividualsPerspective:
    def __init__(self):
        # initialize
        print("Starting estimating individual's perspectives")
        self.use_webcam = False
        self.use_detectron2 = True
        self.save_fig = False
        self.probability_type = 2 # 1: mean of linear distribution, 2: von mises distribution
        if self.use_detectron2:
            self.detectron2 = Detectron2Keypoints()

        # left, top, right, bottom
        self.target = [475, 600, 1200, 0]

        self.detectingAttendedTargets = DetectingAttendedTargets('detecting_attended_targets/model_weights/model_demo.pt', )
        self.gaze360 = Gaze360('gaze360/gaze360_model.pth.tar', )
        self.gazeToFieldOfVision = GazeToFieldOfVision(self.target)
        self.bboxInFieldOfVision = BboxInFieldOfVision(self.target)



    def main(self):
        # Get Data
        if self.use_webcam:
            vc = cv2.VideoCapture(0)
        else:
            vc = cv2.VideoCapture('../../TrainingSet/TwoPersons.m4v') # test.mp4');  TwoPersons.m4v');#

        # Setup plots
        axs = self.setup_plot(2)
        plt.ion()

        # Initialize data / plots
        image_raw, image = self.grab_frame(vc)
        ims = []
        if image_raw is not None:
            ims.append(axs[0].imshow(image));
            ims.append(axs[1].imshow(image));
            self.target[3] = image_raw.shape[0]
            self.gazeToFieldOfVision.set_target(self.target)
            self.bboxInFieldOfVision.set_target(self.target)
        else:
            im = Image.new('RGBA', (1280,720), (0, 0, 0, 255))
            ims.append(axs[0].imshow(im))
            ims.append(axs[1].imshow(im))

        frame_number = 0
        # Analyse images
        while vc.isOpened():
            if self.use_webcam:
                while (True):
                    image_raw, image = self.grab_frame(vc)
                    if cv2.waitKey(1) & 0xFF == ord('q') or image_raw is not None:
                        break
            else:
                for i in range(4): # skip 4 frames
                    image_raw, image = self.grab_frame(vc)
                    if image_raw is None:
                        break;

            if image_raw is None:
                vc.release();
                cv2.destroyAllWindows()
                continue;

            if self.use_detectron2:
                imageShow, face_locations, face_landmarks = self.detectron2.get_keypoints(image_raw)
                imageShow = Image.fromarray(cv2.cvtColor(imageShow, cv2.COLOR_BGR2RGB))
            else:
                face_locations, face_landmarks = self.get_face_locations(image_raw, True, True)
                imageShow = image

            # If faces are detected:
            if len(face_locations) > 0:
                heatmaps, blended = self.plot_detecting_attended_targets(self.detectingAttendedTargets, image, face_locations)
                ims[1].set_data(blended)

                gazes, min, max = self.plot_gaze360(self.gaze360, image, face_locations)
                angles, _ = self.bboxInFieldOfVision.get_bbox_Field_of_vision_angles(gazes, face_landmarks)

                polygons, prob_image = GazeToFieldOfVision.toHeatmap(image, gazes, face_landmarks, min, max, angles)

                heatmapGaze = imageShow.convert("RGBA")
                if self.probability_type == 1:
                    probs = self.gazeToFieldOfVision.get_probabilities(prob_image)
                    # showing Probability image
                    for image in prob_image:
                        heatmapGaze = Image.alpha_composite(heatmapGaze, image.convert("RGBA"))

                ims[0].set_data(heatmapGaze)

                if self.probability_type == 1:
                    while len(probs) > len(axs[0].texts):
                        axs[0].text(10, len(axs[0].texts)*40, "0")
                    t = 0
                    for prob in probs:
                        axs[0].texts[t].set_text("probability for "+str(t)+": "+str(prob))
                        t+=1


                for face in face_locations:
                    polygons.append(
                        patches.Rectangle((face[1], face[0]), face[3] - face[1], face[2] - face[0], linewidth=1,
                                          edgecolor='r', facecolor='none'))
                for ply in polygons:
                    axs[0].add_patch(ply)

                plt.pause(0.0001)
                if self.save_fig:
                    plt.savefig("result/frame"+str(frame_number)+".png")
                    frame_number += 1

                for ply in polygons:
                    ply.remove()
            else:
                ims[0].set_data(imageShow)
                ims[1].set_data(imageShow)
                plt.pause(0.0001)


        plt.ioff()
        plt.show()

    def plot_detecting_attended_targets(self, detectingAttendedTargets, image, face_locations):
        heatmaps = detectingAttendedTargets.getHeatmap(image, face_locations, True)
        map = image.convert("RGBA");
        for heatmap in heatmaps:
            map = Image.alpha_composite(map, heatmap)
        return heatmaps, map
        # blended: Image.alpha_composite(image.convert("RGBA"), heatmaps)
        # blended: Image.blend(image.convert("RGBA"), heatmap, alpha=.5)
        # heatmaps: Image.alpha_composite(Image.new('RGBA', image.size, (0, 0, 0, 0)), heatmap)


    def plot_gaze360(self, gaze360, image, face_locations):
        return gaze360.get_gaze_direction(image, face_locations, True)


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
            face_landmarks = EstimatingIndividualsPerspective.get_eyes(face_recognition.face_landmarks(image))
        if printTime:
            print("Time taken for face recognition: ", time.time() - starttime)
        if get_face_landmarks:
            return face_locations, face_landmarks
        return face_locations, None

    @staticmethod
    def get_eyes(face_landmarks):
        eyes = []
        for landmarks in face_landmarks:
            right = np.mean(landmarks["right_eye"], axis=0)
            left = np.mean(landmarks["left_eye"], axis=0)
            eyes.append(np.asarray(left+(right-left)/2).astype(float))
        return np.array(eyes)

    def setup_plot(self, number_of_axis, titles = ["Gaze360","DAVTV"]):
        fig = plt.figure(figsize=(18, 6))
        #fig.canvas.manager.window.move(0, 0);
        axs = []
        for axis in range(1,number_of_axis+1):
            axs.append(plt.subplot(1, number_of_axis, axis))
            #axs[axis-1].set_axis_off()
            axs[axis - 1].set_title(titles[axis - 1])
        return axs


if __name__ == "__main__":
    EstimatingIndividualsPerspective().main()
