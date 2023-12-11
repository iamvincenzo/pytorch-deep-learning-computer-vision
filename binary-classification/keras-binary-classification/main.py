import random
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import Sequential
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator


# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# parameters
K_FOLDS = 5
PATIENCE = 5
RESIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 200
IMAGE_DIR = "./data/Images"
METADATA_PATH = "./data/metadata.csv"


def custom_preprocess_input(img):
    # apply ResNet50 preprocessing
    img = tf.keras.applications.resnet50.preprocess_input(img)
    
    # Add any additional preprocessing steps if needed
    # For example, you can uncomment and customize the following:
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.random_flip_up_down(img)
    
    return img


def get_generators(df, image_dir, train_indices, valid_indices):
    """
    Create and return image data generators for training and validation sets.

    Args:
        df (pd.DataFrame): input dataframe that contains image names and labels.
        image_dir (str): image paths.
        train_indices (list): contains indicies of training images.
        valid_indices (list): contains indicies of validation images.

    Return:
        training and validation sets.
    """
    # set up the data generators with the custom preprocessing function
    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocess_input,
        # Add other augmentation parameters as needed
        rescale=1.0 / 255,
        # rotation_range=15,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True,
        # fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=custom_preprocess_input,
        rescale=1.0 / 255,
    )

    # generate batches with the custom preprocessing
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df.iloc[train_indices],
        directory=image_dir,
        x_col="filename",
        y_col="label",
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=True,
        class_mode="binary",
        target_size=(RESIZE, RESIZE),
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=df.iloc[valid_indices],
        directory=image_dir,
        x_col="filename",
        y_col="label",
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
        class_mode="binary",
        target_size=(RESIZE, RESIZE),
    )

    return train_generator, validation_generator


def get_model():
    """
    Create and return a binary classification model using a pre-trained ResNet50.

    Return:
        A binary classification model using a pre-trained ResNet50.
    """
    pretrained_model = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(RESIZE, RESIZE, 3),
        pooling="avg",
        weights="imagenet",
    )

    for layer in pretrained_model.layers:
        layer.trainable = False

    binary_model = Sequential()
    binary_model.add(pretrained_model)
    binary_model.add(Flatten())
    binary_model.add(Dense(1024, activation=None)) # "relu"
    binary_model.add(Dense(1, activation="sigmoid"))

    binary_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return binary_model


if __name__ == "__main__":
    # load metadata and shuffle data
    df = shuffle(pd.read_csv(METADATA_PATH), random_state=SEED)
    df["label"] = df["label"].astype(str)

    # initialize KFold
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # perform K-Fold Cross Validation
    for fold, (train_indices, valid_indices) in enumerate(kfold.split(df)):
        print(f"Training on fold {fold + 1}/{K_FOLDS}")

        # create data generators for training and validation
        train_gen, valid_gen = get_generators(df=df, image_dir=IMAGE_DIR, 
                                              train_indices=train_indices, 
                                              valid_indices=valid_indices)

        # create the binary classification model
        binary_model = get_model()

        # add early stopping callback
        early_stopping = EarlyStopping(monitor="val_loss", 
                                       patience=PATIENCE, restore_best_weights=True)

        # add TensorBoard callback
        log_dir = "keras_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # train the model
        history = binary_model.fit(train_gen, validation_data=valid_gen, 
                                   epochs=NUM_EPOCHS, callbacks=[early_stopping, tensorboard_callback])

        # evaluate the model on the validation set
        test_loss, test_accuracy = binary_model.evaluate(valid_gen)
        print(f"Test Accuracy for Fold {fold + 1}: {test_accuracy}")

        # plot training history
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.axis(ymin=0.4, ymax=1)
        plt.grid()
        plt.title(f"Model Accuracy - Fold {fold + 1}")
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend(["train", "validation"])
        plt.show()

        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.grid()
        plt.title(f"Model Loss - Fold {fold + 1}")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend(["train", "validation"])
        plt.show()

    print("K-Fold Cross Validation Complete")
