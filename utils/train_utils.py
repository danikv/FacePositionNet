from keras_preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Model
import numpy as np
import pandas
import keras
import cv2

def custom_loss(ground_truth, prediction):
    A, _ = cv2.Rodrigues(ground_truth, None)
    B, _ = cv2.Rodrigues(prediction, None)
    theta = np.arccos( ( np.trace(A.T @ B) - 1 )/ 2)
    theta = np.rad2deg(theta)
    return theta

def load_images(labels_path):
    datagen = ImageDataGenerator(validation_split=0.4)
    labels = load_labels(labels_path)
    return datagen.flow_from_dataframe(dataframe=labels, x_col="file name", y_col=["rx", "ry", "rz"], class_mode='raw', batch_size=32, target_size=(299,299), subset='training') ,\
        datagen.flow_from_dataframe(dataframe=labels, x_col="file name", y_col=["rx", "ry", "rz"], class_mode='raw', batch_size=64, target_size=(299,299), subset='validation')

def load_labels(labels_path):
    return pandas.read_csv(labels_path)


def create_model():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    x=Dense(512,activation='relu')(x) #dense layer 4
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(3, kernel_initializer='normal')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

