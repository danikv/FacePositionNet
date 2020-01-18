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

df.to_csv(args['output_csv'])