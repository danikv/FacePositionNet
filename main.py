from collections import OrderedDict
import numpy as np
import cv2
import argparse
from utils.preprocess import preprocess_image
from utils.model_loader import ModelLoader
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

model_loader = ModelLoader('model/model3D.mat', 'model/shape_predictor_68_face_landmarks.dat')

image, landmarks, bbox = preprocess_image(args['image'], model_loader)

for i, single_landmark in enumerate(landmarks):
    (x,y,w,h) = bbox[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in landmarks[i]:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
 
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)