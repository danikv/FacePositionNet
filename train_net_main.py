import argparse
from utils.train_utils import load_images, create_model
from keras.callbacks import ModelCheckpoint

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

    filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    STEP_SIZE_TRAIN=images_train_datagen.n//images_train_datagen.batch_size
    STEP_SIZE_VALID=images_train_datagen.n//images_validation_datagen.batch_size
    #STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
    model.fit_generator(generator=images_train_datagen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=images_validation_datagen,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=100,
                        callbacks=callbacks_list,
                        workers=8)
