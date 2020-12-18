"""
face_recognition_videos.py: A simple face recognizer using Python, OpenCV and face-recognition.
"""

__author__ = "S Sathish Babu"
__date__   = "19-12-2020 Saturday 01:00"
__email__  = "bumblebee211196@gmail.com"

import argparse
import pickle

import cv2
import face_recognition
import numpy as np

RESOURCES = 'resources'

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help='Path to the Video file. Points to webcam by default.', default=0)
parser.add_argument('-m', '--model', help='Face detection method, `hog` or `cnn`. Use `cnn` if GPU is available.', default='hog')
args = vars(parser.parse_args())


def detect_face(image):
    locations = face_recognition.face_locations(image, model=args['model'])
    encodings = face_recognition.face_encodings(image, locations)

    for location, encoding in zip(locations, encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        distance = face_recognition.face_distance(known_encodings, encoding)
        match_idx, name = np.argmin(distance), 'Unknown'
        if matches[match_idx]:
            name = known_names[match_idx]
        y1, x2, y2, x1 = location
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y2), (x2, y2 + 25), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, name, (x1 + 7, y2 + 23), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


data = pickle.loads(open(f'{RESOURCES}/encodings', 'rb').read())
known_names, known_encodings = data['names'], data['encodings']

vc = cv2.VideoCapture(args['video'])

while True:
    _, frame = vc.read()
    detect_face(frame)
    cv2.imshow('Output', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
