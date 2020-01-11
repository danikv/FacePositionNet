from utils.threeD_model import FaceModel
import dlib

class ModelLoader:

    def __init__(self, model3d_path, dlib_path):
        self.model3D = FaceModel(model3d_path, 'model3D', False)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_path)