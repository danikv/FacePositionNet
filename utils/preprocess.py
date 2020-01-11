from utils.camera_calibration import estimate_camera, matrix2angle
from utils.ThreeD_Model import FaceModel
import numpy as np
import dlib
import cv2
import os


def _shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=np.float32)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
  # take a bounding predicted by dlib and convert it
  # to the format (x, y, w, h) as we would normally do
  # with OpenCV
  x = rect.left()
  y = rect.top()
  w = rect.right() - x
  h = rect.bottom() - y

  # return a tuple of (x, y, w, h)
  return (x, y, w, h)

def calculate_landmarks(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

    landmarks = []
    dets, scores, idx = detector.run(img, 1)
    shapes = []
    bboxes = []
    for k, det in enumerate(dets):
        shape = predictor(img, det)
        shapes.append(shape)
        xy = _shape_to_np(shape)
        bboxes.append(rect_to_bb(det))
        landmarks.append(xy)

    landmarks = np.asarray(landmarks, dtype='float32')
    return landmarks, bboxes

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    model3D = FaceModel(os.path.join('model' , 'model3D.mat'), 'model3D', False)
    landmarks, bbox = calculate_landmarks(img)
    for i, single_landmark in enumerate(landmarks):
        proj_matrix, camera_matrix, rmat, tvec = estimate_camera(model3D, single_landmark)
        pose_angle = matrix2angle(rmat)
        print(pose_angle)
        print(tvec)
    return img, landmarks, bbox