from utils.camera_calibration import estimate_camera, matrix2angle
from lxml import objectify
from math import cos, sin, radians
import pandas as pd
import numpy as np
import glob
import cv2
import os


image_number = 0

def crop(image, bbox):
    height, width = image.shape[:2]
    new_height = int(height / 8)
    new_width = int(width / 8)
    return image[max(0, bbox[2] - new_height) : min(bbox[3] + new_height, height - 1), max(0 , bbox[0] - new_width) : min(bbox[1] + new_width, width - 1)]

def read_bbox_from_file(file_path):
    xml_file = open(file_path, "r")
    root = objectify.fromstring(str(xml_file.read()))
    bbox = root['object']['bndbox']
    return (bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax'])

def rotate_and_scale_image(image, angle, scale):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated, m

def translate_image(image, horizontal_translation, vertical_translation):
    height, width, _ = image.shape

    m = np.float32([[1, 0, horizontal_translation], [0, 1, vertical_translation]])

    return cv2.warpAffine(image, m, (width, height)), m

def calculate_landmarks_with_respect_to_matrix(landmarks, matrix):
    new_landmarks = np.ones((landmarks.shape[0], 3))
    new_landmarks[:, :-1] = landmarks
    return np.transpose(np.dot(matrix, np.transpose(new_landmarks)))

def _shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=np.float32)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def calculate_landmarks(img, model_loader):
    landmarks = []
    dets, scores, idx = model_loader.detector.run(img, 1)
    shapes = []
    for k, det in enumerate(dets):
        shape = model_loader.predictor(img, det)
        shapes.append(shape)
        xy = _shape_to_np(shape)
        landmarks.append(xy)

    landmarks = np.asarray(landmarks, dtype='float32')
    return landmarks

def convert_data_to_dict(tvec, pose_angle, image_path):
    record = {}
    record['file name'] = image_path
    record['rx'] = pose_angle[0]
    record['ry'] = pose_angle[1]
    record['rz'] = pose_angle[2]
    record['tx'] = tvec[0][0]
    record['ty'] = tvec[1][0]
    record['tz'] = tvec[2][0]
    return record

def flip_image(image):
    return cv2.flip(image, 1)

def flip_landmarks(image, landmarks):
    _, width = image.shape[:2]
    new_landmarks = []
    for x,y in landmarks:
        new_landmarks.append((width - x, y))
    return np.array(new_landmarks)

def random_horizontal_translation(img):
    _, width = img.shape[:2]
    return width * np.random.uniform(-0.1, 0.1)

def random_vertical_translation(img):
    height, _ = img.shape[:2]
    return height * np.random.uniform(-0.1, 0.1)

def random_scale():
    return np.random.uniform(0.75, 1.25)

def random_rotation():
    return 60 * np.random.normal(0, 1)

def calculate_record(image, image_path, model_loader, landmarks):
    _, _, rmat, tvec = estimate_camera(model_loader.model3D, landmarks)
    pose_angle = matrix2angle(rmat)
    return convert_data_to_dict(tvec, pose_angle, image_path)

def save_image(image_path, image):
    global image_number
    image_number += 1
    new_image_path = os.path.join(image_path, str(image_number) + ".jpg")
    cv2.imwrite(new_image_path, image)
    return new_image_path

def preprocess_image(new_path, image_path, annotation_file_path, model_loader):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return []
    bbox = read_bbox_from_file(annotation_file_path)
    img = crop(img, bbox)
    landmarks = calculate_landmarks(img, model_loader)
    records = []
    for i, single_landmark in enumerate(landmarks):
        new_image_path = save_image(new_path, img)
        records.append(calculate_record(img, new_image_path, model_loader, single_landmark))
        img, traslation_matrix = translate_image(img, random_horizontal_translation(img), random_vertical_translation(img))
        img, rotation_matrix = rotate_and_scale_image(img, random_rotation(), random_scale())
        single_landmark = calculate_landmarks_with_respect_to_matrix(single_landmark, traslation_matrix)
        single_landmark = calculate_landmarks_with_respect_to_matrix(single_landmark, rotation_matrix)
        new_image_path = save_image(new_path, img)
        records.append(calculate_record(img, image_path, model_loader, single_landmark))
        img = flip_image(img)
        single_landmark = flip_landmarks(img, single_landmark)
        new_image_path = save_image(new_path, img)
        records.append(calculate_record(img, image_path, model_loader, single_landmark))
    return records

def preprocess_images(images_folder, new_dataset_path, model_loader):
    df = []
    for img in os.listdir(images_folder):
        if img.endswith(".jpg") or img.endswith(".png"):
            for record in preprocess_image(new_dataset_path , os.path.join(images_folder, img), os.path.join(images_folder, '..', 'labels', img.split('.')[0]+'.xml'), model_loader):
                df.append(record)
    return pd.DataFrame(df)