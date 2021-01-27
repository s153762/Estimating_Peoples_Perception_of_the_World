import os
import json
from PIL import Image
import cv2
import matplotlib
matplotlib.use('macosx')#.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import face_recognition
import time
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from detecting_attended_targets.detecting_attended_targets import DetectingAttendedTargets
from gaze360.gaze360 import Gaze360
from gaze_field_of_vision import GazeToFieldOfVision
from detectron.detecron2_keypoint import Detectron2Keypoints
from bbox_in_field_of_vision import BboxInFieldOfVision
from distribution import Distribution

class EstimatingIndividualsPerspective:
    def __init__(self):
        # initialize
        print("Starting estimating individual's perspectives")
        self.use_webcam = False
        self.use_detectron2 = True
        self.save_fig = False
        self.save_vid = True
        self.plot_frames = False
        self.threshold = 0.25
        self.probability_within_threshold = []
        self.save_probs = []
        self.probability_type = 2 # 1: mean of linear distribution, 2: von mises distribution
        self.skip_initial_frames = 0
        if self.use_detectron2:
            self.detectron2 = Detectron2Keypoints()

        # left, top, right, bottom
        self.target = [475, 600, 1200, 0]

        self.detectingAttendedTargets = DetectingAttendedTargets('detecting_attended_targets/model_weights/model_demo.pt', )
        self.gaze360 = Gaze360('gaze360/gaze360_model.pth.tar', )
        self.gazeToFieldOfVision = GazeToFieldOfVision(self.target)
        self.bboxInFieldOfVision = BboxInFieldOfVision(self.target)
        self.distribution = Distribution()



    def main(self, input, output):
        # Get Data
        if self.use_webcam:
            vc = cv2.VideoCapture(0)
        else:
            vc = cv2.VideoCapture(input)

        # Setup plots
        axs, canvas = self.setup_plot(2)
        if self.plot_frames:
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

        # Skip the initial frames
        for i in range(self.skip_initial_frames):
            image_raw, image = self.grab_frame(vc)
            if image_raw is None:
                break;

        # Analyse images
        first_run = True
        while vc.isOpened():
            if self.use_webcam:
                while (True):
                    image_raw, image = self.grab_frame(vc)
                    if cv2.waitKey(1) & 0xFF == ord('q') or image_raw is not None:
                        break
            else:
                for i in range(1): # skip 2 frames
                    image_raw, image = self.grab_frame(vc)
                    if image_raw is None:
                        break;

            if image_raw is None:
                vc.release();
                out.release();
                cv2.destroyAllWindows()
                continue;

            if self.use_detectron2:
                imageShow, face_locations, face_landmarks = self.detectron2.get_keypoints(image_raw)
                imageShow = Image.fromarray(cv2.cvtColor(imageShow, cv2.COLOR_BGR2RGB))
                imageShow = image
            else:
                face_locations, face_landmarks = self.get_face_locations(image_raw, True, True)
                imageShow = image

            # If faces are detected:
            if len(face_locations) > 0:
                heatmaps, blended = self.plot_detecting_attended_targets(self.detectingAttendedTargets, image, face_locations)
                ims[1].set_data(blended)

                gazes, gazes_10, gazes_90 = self.plot_gaze360(self.gaze360, image, face_locations)
                angles_bbox, opposites = self.bboxInFieldOfVision.get_bbox_angles(gazes, face_landmarks)
                polygons, prob_image = GazeToFieldOfVision.get_probability_heatmap(image, gazes, face_landmarks, gazes_10, gazes_90, angles_bbox)

                heatmapGaze = imageShow.convert("RGBA")
                if self.probability_type == 1:
                    probs = self.gazeToFieldOfVision.get_probabilities(prob_image)
                    # showing Probability image
                    for image in prob_image:
                        heatmapGaze = Image.alpha_composite(heatmapGaze, image.convert("RGBA"))
                elif self.probability_type == 2:
                    probs = []
                    for i in range(len(gazes)):
                        angle_gaze_min, angle_gaze_max = GazeToFieldOfVision.get_min_max_angles(gazes[i], gazes_10[i], gazes_90[i])
                        error_angle = max(angle_gaze_min, angle_gaze_max)
                        self.distribution.vonmises(error_angle)
                        probs.append(self.distribution.target_probability(angles_bbox[i][0],angles_bbox[i][1], opposites[i]))
                        #self.distribution.plot()

                ims[0].set_data(heatmapGaze)

                can_see_target = self.within_threshold(probs)
                self.probability_within_threshold.append(can_see_target)
                self.save_probs.append(probs)
                self.display_total_threshold(axs[0])
                polygons += self.create_face_bbox(face_locations, can_see_target)
                #polygons += self.create_target_bbox(self.target)

                # print probabilities
                while len(probs) > len(axs[0].texts)-1:
                    axs[0].text(10, len(axs[0].texts)*40, "0")

                t = 1
                for prob in probs:
                    axs[0].texts[t].set_text("Probability for "+str(t-1)+": "+"%.2f" % prob)
                    t+=1

                axs[0].legend(handles=polygons)
                for ply in polygons:
                    axs[0].add_patch(ply)

                if self.plot_frames:
                    plt.pause(0.0001)
                if self.save_vid:
                    canvas.draw()
                    # https://stackoverflow.com/questions/42603161/convert-an-image-shown-in-python-into-an-opencv-image
                    # https://python.hotexamples.com/examples/matplotlib.backends.backend_agg/FigureCanvasAgg/tostring_rgb/python-figurecanvasagg-tostring_rgb-method-examples.html
                    buf = canvas.tostring_rgb()
                    if first_run:
                        # Save in video
                        w, h = canvas.get_width_height()
                        out = cv2.VideoWriter(output+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (w, h), True)
                        first_run = False
                    figure = np.array(list(buf),'uint8').reshape(h, w, 3)
                out.write(cv2.cvtColor(figure,cv2.COLOR_RGB2BGR))
                if self.save_fig:
                    plt.savefig("result/frame"+str(frame_number)+".png")
                    frame_number += 1

                for ply in polygons:
                    ply.remove()
            else:
                ims[0].set_data(imageShow)
                ims[1].set_data(imageShow)
                if self.plot_frames:
                    plt.pause(0.0001)

        if self.plot_frames:
            plt.ioff()
            plt.show()
        return self.save_probs

    def create_face_bbox(self, face_locations, can_see_target):
        polygons = []
        for i in range(len(face_locations)):
            color = 'g' if can_see_target[i] else 'r'
            polygons.append(
            patches.Rectangle((face_locations[i,1], face_locations[i,0]),
                              face_locations[i,3] - face_locations[i,1],
                              face_locations[i,2] - face_locations[i,0],
                              linewidth=1, edgecolor=color, facecolor='none',label=i))
        return polygons

    def create_target_bbox(self, target):
        return [patches.Rectangle((target[1], target[0]),
                              target[3] - target[1],
                              target[2] - target[0],
                              linewidth=1, edgecolor='b', facecolor='none', label="Target")]

    def within_threshold(self,probs):
        return np.array(probs)>self.threshold

    def display_total_threshold(self, ax):
        true_values = sum(self.probability_within_threshold)
        total_prob = np.round(true_values/len(self.probability_within_threshold),2)
        if len(ax.texts) <= 0:
            ax.text(10, -10, "0")
        total_text = "Total probability is "+str(total_prob)
        ax.texts[0].set_text(total_text)

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

    def setup_plot(self, number_of_axis, titles = ["DAVTV","Gaze360"]):
        fig = plt.figure(figsize=(18, 6))
        #fig.canvas.manager.window.move(0, 0);
        canvas = FigureCanvasAgg(fig)
        axs = []
        for axis in range(1,number_of_axis+1):
            axs.append(plt.subplot(1, number_of_axis, axis))
            #axs[axis-1].set_axis_off()
            axs[axis - 1].set_title(titles[axis - 1])
        temp = axs[0]
        axs[0] = axs[1]
        axs[1] = temp
        return axs, canvas


if __name__ == "__main__":

    directory = "../Test_data/Redigeret_Test1_Test2"
    directory_output = "../Test_data/Results_Test1_Test2"
    files = os.listdir(directory)
    probs = {}
    i = 0
    for file in files:
        input = directory+"/"+file
        output = directory_output+"/"+os.path.splitext(file)[0]+"-result"
        print("-----------------( "+input+", "+output+" )-----------------"+str(i))
        starttime = time.time()
        probs[os.path.splitext(file)[0]] = EstimatingIndividualsPerspective().main(input, output)
        probs[os.path.splitext(file)[0]+"-time"] = time.time() - starttime
        print("Time taken to estimate gaze369: ", probs[os.path.splitext(file)[0]+"-time"])
        i += 1

    with open(directory_output+'/results_test1_test2.json', 'w') as fp:
        json.dump(probs, fp, sort_keys=True, indent=4)

    # ../../TrainingSet/TwoPersons.m4v');
    # ../../TrainingSet/test.mp4');
    # ../../TrainingSet/frontHeadpose.m4v');
    # '../Test_data/Test1/240/2021-01-26-111654.mp4'
    # '../Test_data/Test2/170/2021-01-26-112002.mp4'

