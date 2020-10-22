from PIL import Image
import cv2
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import face_recognition
import time

from detecting_attended_targets.Detecting_attended_targets import DetectingAttendedTargets
from gaze360.Gaze360 import Gaze360
from combination_method import CombinationMethod

class EstimatingIndividualsPerspective:
    def __init__(self):
        print("Starting estimating individual's perspectives")

    def main(self):
        detectingAttendedTargets = DetectingAttendedTargets('detecting_attended_targets/model_weights/model_demo.pt',)
        gaze360 = Gaze360('gaze360/gaze360_model.pth.tar',)

        axs = self.setup_plot(2)
        plt.ion()

        vc = cv2.VideoCapture('../../TrainingSet/test.mp4');
        image_raw, image = self.grab_frame(vc)
        ims = []
        ims.append(axs[0].imshow(image));
        ims.append(axs[1].imshow(image));

        while vc.isOpened():
            for i in range(4):
                image_raw, image = self.grab_frame(vc)

            if image is None:
                break;

            face_locations = self.get_face_locations(image_raw, True)
            if len(face_locations) > 0:
                heatmap, blended = self.plot_detecting_attended_targets(detectingAttendedTargets, image, face_locations)
                ims[0].set_data(blended)
                ims[1].set_data(heatmap)

                #arrows, heads = self.plot_gaze360(gaze360, image, face_locations)
                gazes, eyes = self.plot_gaze360(gaze360, image, face_locations)
                polygon = CombinationMethod.toHeatmap(image, gazes, eyes)
                for ply in polygon:
                    axs[0].add_patch(ply)
                #for arrow in arrows+heads:
                #    axs[0].add_patch(arrow)

                plt.pause(0.0001)
                #for arrow in arrows+heads:
                #    arrow.remove()
                for ply in polygon:
                    ply.remove()
            else:
                ims[0].set_data(image)
                ims[1].set_data(Image.new('RGBA', image.size, (0, 0, 0, 255)))
                plt.pause(0.0001)


        plt.ioff()
        plt.show()

    def plot_detecting_attended_targets(self, detectingAttendedTargets, image, face_locations):
        heatmap = detectingAttendedTargets.getHeatmap(image, face_locations, True)
        return Image.alpha_composite(Image.new('RGBA', image.size, (0, 0, 0, 255)), heatmap), Image.alpha_composite(image.convert("RGBA"), heatmap) #Image.blend(image.convert("RGBA"), heatmap, alpha=.5)


    def plot_gaze360(self, gaze360, image, face_locations):
        arrows = gaze360.getArrows(image, face_locations, True, False)
        return arrows


    def grab_frame(self, vc):
        ret, image_raw = vc.read()
        if image_raw is None:
            return None
        frame = Image.fromarray(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))
        return image_raw, frame.convert('RGB')

    def get_face_locations(self, image, printTime):
        starttime = None
        if printTime:
            starttime = time.time()
        face_locations = face_recognition.face_locations(image)
        if printTime:
            print("Time taken for face recognition: ", time.time() - starttime)
        return face_locations

    def setup_plot(self, number_of_axis):
        fig = plt.figure(figsize=(10, 5))  # 15,10))
        fig.canvas.manager.window.move(0, 0);
        axs = []
        for axis in range(1,number_of_axis+1):
            axs.append(plt.subplot(1, number_of_axis, axis))
            axs[axis-1].set_axis_off()

        return axs




if __name__ == "__main__":
    EstimatingIndividualsPerspective().main()
