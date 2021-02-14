import face_recognition
import uuid
import numpy as np

# recognizes the people in the video.
class IdentifyPeople:

    def __init__(self):
        self.people = {}

    def identify_faces(self,face_locations,face_landmarks, image, only_one_person):
        # If no other person is expected.
        if only_one_person:
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
        return identified, self.people

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