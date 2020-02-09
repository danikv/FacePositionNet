from keras_preprocessing.image import ImageDataGenerator
from keras.applications import ResNet101V2, InceptionV3, InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from scipy.spatial.transform import Rotation as R
from keras.models import load_model
import keras.backend as K
from keras import Model
import numpy as np
import pandas
import keras
import cv2

def calculate_average_error(df, model):
    results = []
    for index, row in df.iterrows():
        image_path = row['file name']
        x = load_image_as_array(image_path)
        prediction = model.predict(x)
        ground_truth = np.asarray([row['rx'], row['ry'], row['rz']], dtype=np.float)
        results.append(custom_loss(ground_truth, prediction))
    return sum(results) / len(results)

def custom_loss(y_true, y_pred):
    A, _ = cv2.Rodrigues(y_true, None)
    B, _ = cv2.Rodrigues(y_pred, None)
    theta = np.arccos( ( np.trace(A.T @ B) - 1 )/ 2)
    theta = np.rad2deg(theta)
    return theta if 180 - theta > 10 else 180-theta
    #return K.mean(K.square(y_true - y_pred))

def load_images(labels_path):
    datagen = ImageDataGenerator()
    labels = load_labels(labels_path)
    return datagen.flow_from_dataframe(dataframe=labels, x_col="file name", y_col=["rx", "ry", "rz"], class_mode='raw', batch_size=64, target_size=(299,299))

def load_labels(labels_path):
    return pandas.read_csv(labels_path)

def load_test_csv(path):
    return pandas.read_csv(path)

def load_image_as_array(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(299,299))
    x = keras.preprocessing.image.img_to_array(img)
    return x.reshape(1,299,299,3).astype('float')

def create_model():
    # create the base pre-trained model
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1024,activation='relu')(x) #dense layer 2
    x = Dropout(0.2)(x)
    x = Dense(512,activation='relu')(x) #dense layer 3

    x = Dropout(0.3)(x)

    #x = GlobalAveragePooling2D()(x)
    #x = Dense(1024,activation='relu')(x) #dense layer 4

    predictions = Dense(3, kernel_initializer='normal')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-60:]:
        layer.trainable = True
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def load_model_from_file(path):
    model = load_model(path)
    return model

def predict_on_image(model, image_path):
    image = load_image_as_array(image_path)
    return model.predict(image)