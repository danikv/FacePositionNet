from utils.train_utils import load_images, create_model, load_test_csv, calculate_average_error, predict_on_image, load_model_from_file
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
from keras import backend as K
import pandas as pd

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_csv", required=False,
        help="path to input image")
    ap.add_argument("-iv", "--input_validation_csv", required=False,
        help="path to input image")
    ap.add_argument("-ip", "--image_path", required=False,
        help="path of the image to run on")
    ap.add_argument("-m", "--model_path", required=False,
        help="model path")
    ap.add_argument("-md", "--model_dir", required=False,
        help="model dir path for validation checking")
    ap.add_argument("-id", "--images_dir", required=False,
        help="images dir to test on")
    ap.add_argument("-oc", "--output_csv", required=False,
        help="output csv file")
    args = ap.parse_args()

    if args.input_csv and args.input_validation_csv:
        images_train_datagen = load_images(args.input_csv)
        images_validation_datagen = load_images(args.input_validation_csv)

        model = create_model()

        model.load_weights(os.path.abspath('inception-resnet-v2-60layers-53.40-338.31.hdf5'))

        filepath="inception-resnet-v2-60layers-{val_loss:.2f}-{val_loss2:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        STEP_SIZE_TRAIN=images_train_datagen.n//images_train_datagen.batch_size
        STEP_SIZE_VALID=images_validation_datagen.n//images_validation_datagen.batch_size
        #STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
        history_callback = model.fit_generator(generator=images_train_datagen,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=images_validation_datagen,
                            validation_steps=STEP_SIZE_VALID,
                            epochs=150,
                            callbacks=callbacks_list,
                            workers=8)
        numpy_loss_history = np.array(history_callback.history["loss"], dtype="float32")
        numpy_val_history = np.array(history_callback.history["val_loss"], dtype="float32")
        loss = np.array([(numpy_loss_history[i], numpy_val_history[i]) for i in range(len(numpy_loss_history))])
        np.savetxt("loss_history_inception_resnet_v2_60layers.txt", loss, delimiter=",")
    elif args.image_path and args.model_path:
        model = load_model_from_file(args.model_path)
        print(predict_on_image(model, args.image_path)[0])
    elif args.images_dir and args.model_path and args.output_csv:
        model = load_model_from_file(args.model_path)
        df = []
        for img in os.listdir(args.images_dir):
            if img.endswith(".jpg") or img.endswith(".png"):
                prediction = predict_on_image(model, os.path.join(args.images_dir, img))[0]
                df.append({'file name': img, 'rx': prediction[0], 'ry': prediction[1], 'rz': prediction[2]})
        pd.DataFrame(df).to_csv(args.output_csv)
    elif args.model_dir:
        df = load_test_csv('valid_set/validation_set.csv')
        df2 = load_test_csv('new_valid_set/valid_set2.csv')
        best_model = ""
        best_error = 100000000
        for filename in os.listdir(args.model_dir):
            if filename.endswith(".hdf5"):
                K.clear_session()
                model = load_model_from_file(os.path.join(args.model_dir, filename))
                new_error = (calculate_average_error(df , model) + calculate_average_error(df2 , model)) / 2
                print(new_error)
                if new_error < best_error :
                    #save model
                    best_model = filename
                    best_error = new_error
        print(best_model)
        print(best_error)
    else :
        ap.print_help(sys.stderr)