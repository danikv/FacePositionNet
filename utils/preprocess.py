from utils.camera_calibration import estimate_camera, matrix2angle
from scipy.spatial.transform import Rotation as R
from math import cos, sin, radians
from lxml import objectify
import pandas as pd
import numpy as np
import glob
import cv2
import os

image_number = 0

def split_dataset(dataset):
    msk = np.random.rand(len(dataset)) < 0.7
    train = dataset[msk]
    validation = dataset[~msk]
    return train, validation

def load_pts_file(pts_path):
    try :
        with open(pts_path) as f:
            rows = [rows.strip() for rows in f]
    except OSError:
        return None
    
    """Use the curly braces to find the start and end of the point data""" 
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = np.asarray([np.asarray([float(point) for point in coords]) for coords in coords_set])
    return np.array([points])

def crop(image, bbox):
    height, width = image.shape[:2]
    new_height = int(height / 8)
    new_width = int(width / 8)
    rotation = (max(0 , bbox[0] - new_width), max(0, bbox[2] - new_height))
    return rotation, image[max(0, bbox[2] - new_height) : min(bbox[3] + new_height, height), max(0 , bbox[0] - new_width) : min(bbox[1] + new_width, width)]

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

def calculate_new_landmark_with_rotation(landmark, rotation):
    coords = []
    for x,y in landmark:
        coords.append((x + rotation[0], y + rotation[1]))
    return np.asarray(coords, dtype='float32')

def rotate_landmarks_before_crop(landmarks, rotation):
    new_landmarks = []
    for landmark in landmarks:
        new_landmarks.append(calculate_new_landmark_with_rotation(landmark, rotation))
    return np.asarray(new_landmarks, dtype='float32')

def convert_data_to_dict(tvec, pose_angle, image_path):
    record = {}
    rot_vec = R.from_euler('xyz', pose_angle).as_rotvec()
    record['file name'] = image_path
    record['rx'] = rot_vec[0]
    record['ry'] = rot_vec[1]
    record['rz'] = rot_vec[2]
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
    return 30 * np.random.normal(0, 1)

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

def show_image_and_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

def transform_landmarks(landmarks, new_path, img, model_loader):
    records = []
    if len(landmarks) <= 0:
        return records
    single_landmark = landmarks[0]
    new_image_path = save_image(new_path, img)
    records.append(calculate_record(img, new_image_path, model_loader, single_landmark))
    img, traslation_matrix = translate_image(img, random_horizontal_translation(img), random_vertical_translation(img))
    img, rotation_matrix = rotate_and_scale_image(img, random_rotation(), random_scale())
    single_landmark = calculate_landmarks_with_respect_to_matrix(single_landmark, traslation_matrix)
    single_landmark = calculate_landmarks_with_respect_to_matrix(single_landmark, rotation_matrix)
    new_image_path = save_image(new_path, img)
    records.append(calculate_record(img, new_image_path, model_loader, single_landmark))
    img = flip_image(img)
    single_landmark = flip_landmarks(img, single_landmark)
    new_image_path = save_image(new_path, img)
    records.append(calculate_record(img, new_image_path, model_loader, single_landmark))
    return records

def preprocess_image(new_path, images_folder, image_name, model_loader):
    image_path = os.path.join(images_folder, image_name)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    annotation_file_path = os.path.join(images_folder, '..', 'labels', image_name.split('.')[0]+'.xml')
    if img is None:
        return []
    bbox = read_bbox_from_file(annotation_file_path)
    rotation, cropped_img = crop(img, bbox)
    landmarks = calculate_landmarks(cropped_img, model_loader)
    landmarks = rotate_landmarks_before_crop(landmarks, rotation)
    return transform_landmarks(landmarks, new_path, img, model_loader)

def preprocess_validation_image(new_path, images_folder, image_name, model_loader):
    image_path = os.path.join(images_folder, image_name)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    pts_file_path = os.path.join(images_folder, image_name.split('.')[0] + '.pts')
    if img is None:
        return []
    landmarks = load_pts_file(pts_file_path)
    if landmarks is None:
        return []
    return transform_landmarks(landmarks, new_path, img, model_loader)

def preprocess_images(images_folder, new_dataset_path, model_loader, process_function):
    df = []
    for img in os.listdir(images_folder):
        if img.endswith(".jpg") or img.endswith(".png"):
            for record in process_function(new_dataset_path , images_folder, img, model_loader):
                df.append(record)
    return pd.DataFrame(df)