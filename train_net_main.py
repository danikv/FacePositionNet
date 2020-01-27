from utils.train_utils import load_images, create_model, load_test_csv, custom_loss, load_image_as_array
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import argparse
import os

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_csv", required=True,
        help="path to input image")
    ap.add_argument("-iv", "--input_validation_csv", required=True,
        help="path to input image")
    args = ap.parse_args()

    images_train_datagen = load_images(args.input_csv)
    images_validation_datagen = load_images(args.input_validation_csv)

    model = create_model()

    #model.load_weights(os.path.abspath('incerptionv3-02-1.23.hdf5'))

    filepath="incerptionv3-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    STEP_SIZE_TRAIN=images_train_datagen.n//images_train_datagen.batch_size
    STEP_SIZE_VALID=images_validation_datagen.n//images_validation_datagen.batch_size
    #STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    model.fit_generator(generator=images_train_datagen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=images_validation_datagen,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=100,
                        callbacks=callbacks_list,
                        workers=8)

    df = load_test_csv('valid_set/validation_set.csv')
    results = []
    model_results = []
    ground_truths = []

    for index, row in df.iterrows():
        image_path = row['file name']
        x = load_image_as_array(image_path)
        prediction = model.predict(x)
        ground_truth = np.asarray([row['rx'], row['ry'], row['rz']], dtype=np.float)
        results.append(custom_loss(ground_truth, prediction))
        print(results[-1])
        model_results.append(prediction)
        ground_truths.append(ground_truth)
    

    print('average error : ' + str(sum(results) / len(results)))

    with open('ground_truth.txt', 'w+') as f:
        for item in ground_truths:
            f.write("%s\n" % item)

    with open('model_results.txt', 'w+') as f:
        for item in model_results:
            f.write("%s\n" % item)