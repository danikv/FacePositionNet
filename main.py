from collections import OrderedDict
import numpy as np
import cv2
import argparse
from utils.preprocess import preprocess_images, preprocess_image, preprocess_validation_image, split_dataset
from utils.model_loader import ModelLoader
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input images")
ap.add_argument("-iv", "--images_validation", required=True,
	help="path to input validation images")
ap.add_argument("-o", "--output_csv", required=True,
	help="path to output the training csv")
ap.add_argument("-ov", "--output_validation_csv", required=True,
	help="path to output the validation csv")
ap.add_argument("-of", "--output_folder", required=True,
	help="path to output transformed images")
args = ap.parse_args()

model_loader = ModelLoader('model/model3D.mat', 'model/shape_predictor_68_face_landmarks.dat')

df = preprocess_images(args.images, args.output_folder, model_loader, preprocess_image)

train, validation = split_dataset(df)

df_validation = preprocess_images(args.images_validation, args.output_folder, model_loader, preprocess_validation_image)

train.to_csv(args.output_csv)

df_validation.append(validation)

df_validation.to_csv(args.output_validation_csv)

