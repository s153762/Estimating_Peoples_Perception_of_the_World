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
import uuid


from detecting_attended_targets.detecting_attended_targets import DetectingAttendedTargets
from gaze360.gaze360 import Gaze360
from gaze_field_of_vision import GazeToFieldOfVision
from detectron.detecron2_keypoint import Detectron2Keypoints
from bbox_in_field_of_vision import BboxInFieldOfVision
from distribution import Distribution

class EstimatingIndividualsPerspective:
    def __init__(self):
        # initialize
        self.use_webcam = False
        self.use_detectron2 = True
        self.use_detecting_attention = False
        self.save_fig = False
        self.save_vid = True
        self.plot_frames = False
        self.threshold = 0.25
        self.probability_within_threshold = {}
        self.save_probs = {}
        self.only_one_person = False
        self.probability_type = 3 # 1: mean of linear distribution, 2: von mises distribution, 3: front facing
        self.skip_initial_frames = 0
        self.people = {}
        if self.use_detectron2:
            self.detectron2 = Detectron2Keypoints()
            self.show_detectron = True

        # left, top, right, bottom
        self.target = [400, 650, 1270, 719]
        self.detectingAttendedTargets = DetectingAttendedTargets('detecting_attended_targets/model_weights/model_demo.pt', )
        self.gaze360 = Gaze360('gaze360/gaze360_model.pth.tar', )
        self.gazeToFieldOfVision = GazeToFieldOfVision(self.target)
        self.bboxInFieldOfVision = BboxInFieldOfVision(self.target)
        self.distribution = Distribution()

    # Getting the data from webcam or video, initializing the plots and skipping initial frames
    def setup_init_images(self, input_dir, axs):
        # Get Data
        if self.use_webcam:
            vc = cv2.VideoCapture(0)
        else:
            vc = cv2.VideoCapture(input_dir)

        # Initialize data / plots
        image_raw, image = self.grab_frame(vc)
        ims = []
        if image_raw is not None:
            ims.append(axs[0].imshow(image));
            ims.append(axs[1].imshow(image));
            self.target[3] = image_raw.shape[0]
        else:
            im = Image.new('RGBA', (1280, 720), (0, 0, 0, 255))
            ims.append(axs[0].imshow(im))
            ims.append(axs[1].imshow(im))

        # Skip the initial frames
        for i in range(self.skip_initial_frames):
            image_raw, image = self.grab_frame(vc)
            if image_raw is None:
                break;
        return vc, ims

    # Extracting new image from webcam or video
    def extract_new_image(self, vc, skip_frames = 2):
        if self.use_webcam:
            while (True):
                image_raw, image = self.grab_frame(vc)
                if cv2.waitKey(1) & 0xFF == ord('q') or image_raw is not None:
                    break
        else:
            for i in range(skip_frames):  # skip 2 frames
                image_raw, image = self.grab_frame(vc)
                if image_raw is None:
                    break;
        return image_raw, image


    def extract_features(self, image_raw, image):
        # Extract people's features
        imageShow = image
        if self.use_detectron2:
            imageShow, face_locations, face_landmarks = self.detectron2.get_keypoints(image_raw)
            imageShow = image if not self.show_detectron else Image.fromarray(
                cv2.cvtColor(imageShow, cv2.COLOR_BGR2RGB))
        else:
            face_locations, face_landmarks = self.get_face_locations(image_raw, True, False)

        # Check if features were found
        if len(face_locations) <= 0 or len(face_landmarks) <= 0:
            return imageShow, {}, {}

        # Sort face locations so first person seen is 0, second is 1, always.
        face_ids = self.identify_faces(face_locations, face_landmarks, image)
        face_locations = {k: v[1] for k, v in self.people.items() if k in face_ids}
        eyes = {k: v[2] for k, v in self.people.items() if k in face_ids}
        return imageShow, face_locations, eyes


    def calculate_gaze360_probabilities(self, imageShow, prob_image, gazes, gazes_10, gazes_90, angles_bbox, opposites, face_locations, frame_number):
        probs = {}
        heatmapGaze = imageShow.convert("RGBA")
        if self.probability_type == 1:
            probs = self.gazeToFieldOfVision.get_probabilities(prob_image)
            # showing Probability image
            for image in prob_image:
                heatmapGaze = Image.alpha_composite(heatmapGaze, prob_image[image].convert("RGBA"))

        elif self.probability_type == 2:
            for k in gazes.keys():
                angle_gaze_min, angle_gaze_max = GazeToFieldOfVision.get_min_max_angles(gazes[k], gazes_10[k], gazes_90[k])
                error_angle = max(angle_gaze_min, angle_gaze_max)
                self.distribution.vonmises(error_angle)
                probs[k] = self.distribution.target_probability(angles_bbox[k][0], angles_bbox[k][1], opposites[k])
                # self.distribution.plot()

        elif self.probability_type == 3:
            for k in gazes.keys():
                # left, top, right, bottom
                head = imageShow.crop((face_locations[k][3],face_locations[k][0],face_locations[k][1],face_locations[k][2]))
                head = np.asarray(head)
                location = face_recognition.face_landmarks(head)
                probs[k] = 1 if len(location) >= 1 else 0

        # Update saved probabilities
        can_see_target = self.within_threshold(probs)
        for k in probs.keys():
            if k not in self.probability_within_threshold.keys():
                self.probability_within_threshold[k] = []
                self.save_probs[k] = []
            self.probability_within_threshold[k].append(can_see_target[k])
            self.save_probs[k].append({"probs": probs[k], "Frame": frame_number})

        return probs, heatmapGaze

    def create_face_bbox(self, face_locations, can_see_target):
        polygons = []
        for k in face_locations.keys():
            color = 'g' if can_see_target[k][-1] else 'r'
            polygons.append(
            patches.Rectangle((face_locations[k][1], face_locations[k][0]),
                              face_locations[k][3] - face_locations[k][1],
                              face_locations[k][2] - face_locations[k][0],
                              linewidth=1, edgecolor=color, facecolor='none',label=str(k)[:5]))
        return polygons

    def create_target_bbox(self, target):
        # left, top, right, bottom
        return [patches.Rectangle((target[0], target[1]),
                              target[2] - target[0],
                              target[3] - target[1],
                              linewidth=0.5, edgecolor='b', facecolor='none', label="Target")]

    def within_threshold(self,probs):
        result = {}
        for k,v in probs.items():
            result[k] = v>self.threshold
        return result

    def display_total_threshold(self, ax):
        output_string = "Average probability is\n"
        for k,v in self.probability_within_threshold.items():
            true_values = sum(list(v))
            total_prob = np.round(true_values/len(v),2)
            output_string += str(k)[:5]+"="+str(total_prob)+"%, \n"

        if len(ax.texts) <= 0:
            ax.text(10, -10, "0")
        ax.texts[0].set_text(output_string)

    def plot_detecting_attended_targets(self, detectingAttendedTargets, image, face_locations):
        heatmaps = detectingAttendedTargets.getHeatmap(image, face_locations, False)
        map = image.convert("RGBA");
        for heatmap in heatmaps:
            map = Image.alpha_composite(map, heatmap)
        return heatmaps, map
        # blended: Image.alpha_composite(image.convert("RGBA"), heatmaps)
        # blended: Image.blend(image.convert("RGBA"), heatmap, alpha=.5)
        # heatmaps: Image.alpha_composite(Image.new('RGBA', image.size, (0, 0, 0, 0)), heatmap)


    def plot_gaze360(self, gaze360, image, face_locations):
        return gaze360.get_gaze_direction(image, face_locations, False)


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
        if not self.use_detecting_attention:
            titles[0] = "Detectron2"
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

    def identify_faces(self,face_locations,face_landmarks, image):
        # If no other person is expected.
        if self.only_one_person:
            self.people["only_one"] = [0,face_locations[0],face_landmarks[0]]
            return ["only_one"]

        # Sort face locations so first person seen is 0, second is 1, always.
        identified = self.add_faces(face_locations,face_landmarks, image)
        for k in self.people.copy():
            if k not in identified:
                if len(self.people[k]) < 4:
                    self.people[k].append(0)
                elif self.people[k][3] >= 50:
                    self.people.pop(k, None)
                else:
                    self.people[k][3]+=1
        return identified

    def add_faces(self, faces, eyes, image):
        identified = {}
        not_identified = []
        encodings = face_recognition.face_encodings(np.array(image), known_face_locations=faces)

        # For all faces check both location and recognition fits
        for i in range(len(encodings)):
            # Is the face is seen in before
            keys_people = self.recognition_identified(encodings[i], {k:self.people[k][0] for k in self.people.keys() if k not in identified})
            # Are any previous faces close?
            closest_key = self.location_identified(eyes[i], faces[i], {k:self.people[k][2] for k in self.people.keys() if k not in identified})

            # One or multiple matches
            if closest_key in keys_people:
                identified[closest_key] = [self.people[closest_key][0], faces[i], eyes[i]]
            # No face recognition match - check second round
            else:
                not_identified.append([encodings[i], faces[i], eyes[i]])

        identified_keys = list(identified.keys())
        if len(identified_keys) == len(faces):
            # save results
            for k,v in identified.items():
                self.people[k] = v
            return identified_keys

        # see if other people might match
        for person in not_identified:
            # check is a face was close but not a match
            find_best_location = {k:self.people[k][2] for k in self.people.keys() if k not in identified_keys}
            key = self.location_identified(person[2], person[1], find_best_location)
            # None are close so create new
            if key is None:
                recognize = self.recognition_identified(encodings[i], {k:self.people[k][0] for k in self.people.keys() if k not in identified_keys})
                if len(recognize) > 1:
                    key = recognize[0]
                else:
                    key = uuid.uuid4()
            identified[key] = person
            identified_keys.append(key)

        for k,v in identified.items():
            self.people[k] = v
        return identified_keys

    def recognition_identified(self, encoding, previous_encodings):
        if len(previous_encodings) == 0:
            return []
        keys, values = zip(*previous_encodings.items())
        result = face_recognition.compare_faces(values, encoding)
        return np.array(keys)[result]

    def location_identified(self, eye_location, face_bbox, known_previous):
        min_movement = ((face_bbox[1] - face_bbox[3])+(face_bbox[2] - face_bbox[0]))/3
        minimum = [None,min_movement]
        for k,v in known_previous.items():
            previous = known_previous[k]
            diff = sum([np.abs(previous[i]-eye_location[i]) for i in range(len(eye_location))])
            if diff < minimum[1]:
                minimum = [k, diff]
        return minimum[0]

    def main(self, input_dir, output):
        # Setup plots
        axs, canvas = self.setup_plot(2)
        if self.plot_frames:
            plt.ion()
        vc, ims = self.setup_init_images(input_dir, axs)

        # Analyse images
        first_run = True
        frame_number = 0
        while vc.isOpened():
            image_raw, image = self.extract_new_image(vc, 2)

            # If no new image, end analysis
            if image_raw is None:
                vc.release();
                out.release();
                cv2.destroyAllWindows()
                continue;

            # Extract people's features
            imageShow, face_locations, eyes = self.extract_features(image_raw, image)

            # If no faces are detected:
            if len(face_locations) <= 0:
                ims[0].set_data(imageShow)
                ims[1].set_data(imageShow)
                if self.plot_frames:
                    plt.pause(0.0001)
                frame_number+=1
                continue

            # Detecting Attended Targets
            ims[1].set_data(imageShow)
            if self.use_detecting_attention:
                heatmaps, blended = self.plot_detecting_attended_targets(self.detectingAttendedTargets, image, face_locations)
                ims[1].set_data(blended)

            # Gaze360
            gazes, gazes_10, gazes_90 = self.plot_gaze360(self.gaze360, image, face_locations)
            angles_bbox, opposites = self.bboxInFieldOfVision.get_bbox_angles(gazes, eyes)
            polygons, prob_image = GazeToFieldOfVision.get_probability_heatmap(image, gazes, eyes, gazes_10, gazes_90, angles_bbox)

            # Calculating the probabilities and update self.probability_within_threshold and self.save_probs
            probs, heatmapGaze = self.calculate_gaze360_probabilities(image, prob_image, gazes, gazes_10, gazes_90, angles_bbox, opposites, face_locations, frame_number)
            ims[0].set_data(heatmapGaze)

            # Print average probabilities
            self.display_total_threshold(axs[0]) # average for frames

            # Add text per person
            while len(probs) > len(axs[0].texts)-1:
                axs[0].text(0, 0, 0)

            # Print individual text per person
            keys = list(probs.keys())
            for t in range(1,len(axs[0].texts)):
                text = ""
                if len(keys) >= t:
                    k = keys[t-1]
                    text = str(k)[:5]+": %.2f" % probs[k]+"%"
                    axs[0].texts[t].set_y(face_locations[k][0])
                    axs[0].texts[t].set_x(face_locations[k][3])
                axs[0].texts[t].set_text(text)

            # Add polygons to gaze360 image
            head_bbox = self.create_face_bbox(face_locations, self.probability_within_threshold)
            polygons += head_bbox
            polygons += self.create_target_bbox(self.target)
            for ply in polygons:
                axs[0].add_patch(ply)
            #axs[0].legend(handles=head_bbox)

            if self.plot_frames:
                plt.pause(0.0001)
            if self.save_vid:
                # https://stackoverflow.com/questions/42603161/convert-an-image-shown-in-python-into-an-opencv-image
                # https://python.hotexamples.com/examples/matplotlib.backends.backend_agg/FigureCanvasAgg/tostring_rgb/python-figurecanvasagg-tostring_rgb-method-examples.html
                canvas.draw()
                buf = canvas.tostring_rgb()
                w, h = canvas.get_width_height()
                if first_run:
                    out = cv2.VideoWriter(output+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w, h), True)
                    first_run = False
                figure = np.array(list(buf),'uint8').reshape(h, w, 3)
                out.write(cv2.cvtColor(figure,cv2.COLOR_RGB2BGR))
            if self.save_fig:
                plt.savefig("result/frame"+str(frame_number)+".png")

            if frame_number % 30 == 0:
                print()
            print(str(frame_number), end=", ")
            frame_number += 1

            for ply in polygons:
                ply.remove()

        # End of video
        if self.plot_frames:
            plt.ioff()
        print()
        return {str(k):v for k,v in self.save_probs.items()}, self.gaze360.gaze360_time, self.detectingAttendedTargets.detectingAttendedTargets_time, self.distribution.distribution_time

if __name__ == "__main__":
    directory = "../Test_data/Test1_Test2/Opdelt"
    directory_output = "../Test_data/Test1_Test2/Opdelt_face_Result"
    files = os.listdir(directory)
    probs = {}
    i = 0
    for file in files:
        if ".DS_Store" in file:
            continue
        input = directory+"/"+file
        output = directory_output+"/"+os.path.splitext(file)[0]+"-result"
        print("-----------------( "+input+", "+output+" )-----------------"+str(i))
        starttime = time.time()
        key = os.path.splitext(file)[0]
        probs[key], gaze360_time, detectingAttendedTargets_time, distribution_time = EstimatingIndividualsPerspective().main(input, output)
        probs[key+"-time"] = time.time() - starttime
        probs[key+"-gaze360_time"] = gaze360_time
        probs[key+"-detectingAttendedTargets_time"] = detectingAttendedTargets_time
        probs[key+"-distribution_time"] = distribution_time
        print("Time taken to estimate perception: ", probs[os.path.splitext(file)[0]+"-time"])
        i += 1

        with open(directory_output+'/results_test1_test2.json', 'w') as fp:
            json.dump(probs, fp, indent=4)

    # ../../TrainingSet/TwoPersons.m4v');
    # ../../TrainingSet/test.mp4');
    # ../../TrainingSet/frontHeadpose.m4v');
    # '../Test_data/Test1/240/2021-01-26-111654.mp4'
    # '../Test_data/Test2/170/2021-01-26-112002.mp4'

