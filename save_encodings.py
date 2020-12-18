"""
save_encodings.py: Creates the encodings of the image dataset and saves it to a custom location.
"""

__author__ = "S Sathish Babu"
__date__   = "19-12-2020 Saturday 01:00"
__email__  = "bumblebee211196@gmail.com"


import argparse
import os
import pickle

import cv2
import face_recognition

RESOURCES = 'resources'

parser = argparse.ArgumentParser('FaceRecognition Application - Face encodings gne')
parser.add_argument('-o', '--output-file', help="Location to save the encodings", default=f'{RESOURCES}/encodings')
parser.add_argument('-m', '--model', help='Face detection method, `hog` or `cnn`. Use `cnn` if GPU is available.', default='hog')
args = vars(parser.parse_args())

names = []
encodings = []

for name in os.listdir(RESOURCES):
    for file in os.listdir(f'{RESOURCES}/{name}'):
        image = cv2.imread(f'{RESOURCES}/{name}/{file}')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(image_rgb, model=args['model'])  # Use model="hog" if no GPU is available
        encodings_ = face_recognition.face_encodings(image_rgb, locations)

        for encoding in encodings_:
            encodings.append(encoding)
            names.append(name)

data = {"names": names, "encodings": encodings}
with open(args['output_file'], 'wb') as fp:
    pickle.dump(data, fp)
