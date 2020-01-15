from collections import OrderedDict
import numpy as np
import cv2
import argparse
from utils.preprocess import preprocess_images, calculate_landmarks, flip_landmarks, translate_image, rotate_and_scale_image, calculate_landmarks_with_respect_to_matrix
from utils.model_loader import ModelLoader
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-o", "--output_csv", required=True,
	help="path to output the csv")
ap.add_argument("-of", "--output_folder", required=True,
	help="path to output the csv")
args = vars(ap.parse_args())

model_loader = ModelLoader('model/model3D.mat', 'model/shape_predictor_68_face_landmarks.dat')

df = preprocess_images(args['image'], args['output_folder'], model_loader)
#np.savetxt('struct_array.csv', df, delimiter=',', header='file name,rx,ry,ry,tx,ty,tz')

df.to_csv(args['output_csv'])

'''img = cv2.imread(args['image'], cv2.IMREAD_COLOR)
landmarks, bbox = calculate_landmarks(img, model_loader)
img, traslation_matrix = translate_image(img, 10, 20)
img, rotation_matrix = rotate_and_scale_image(img, 15, 0.9)
'''
#image = np.flip(image, 1)
#landmarks = np.flip(landmarks, 1)
'''image = cv2.flip(img, 1)

for i, landmark in enumerate(landmarks):
    landmark = calculate_landmarks_with_respect_to_matrix(landmark, traslation_matrix)
    landmark = calculate_landmarks_with_respect_to_matrix(landmark, rotation_matrix)
    landmark = flip_landmarks(image, landmark)

    #for i, single_landmark in enumerate(landmarks):
    (x,y,w,h) = bbox[0]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in landmark:
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
 
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)'''

