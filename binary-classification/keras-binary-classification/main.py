import random
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense, Flatten


# reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

K_FOLDS = 5
PATIENCE = 5
RESIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 200


def get_generators(df, image_dir, train_indices, valid_indices):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_dataframe(
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

    validation_generator = datagen.flow_from_dataframe(
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
    pretrained_model = tf.keras.applications.ResNet50(include_top=False, 
                                                      input_shape=(RESIZE, RESIZE, 3), 
                                                      pooling="avg", weights="imagenet")

    for layer in pretrained_model.layers:
        layer.trainable = False

    binary_model = Sequential()
    binary_model.add(pretrained_model)
    binary_model.add(Flatten())
    binary_model.add(Dense(16, activation="relu"))
    binary_model.add(Dense(1, activation="sigmoid"))

    binary_model.compile(optimizer=Adam(learning_rate=0.001), 
                         loss="binary_crossentropy", metrics=["accuracy"])
    
    return binary_model

if __name__ == "__main__":
    metadata_path = "./data/metadata.csv"
    df = shuffle(pd.read_csv(metadata_path), random_state=SEED)
    df["label"] = df["label"].astype(str)
    image_dir = "./data/Images"

    # Initialize KFold
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_indices, valid_indices) in enumerate(kfold.split(df)):
        print(f"Training on fold {fold + 1}/{K_FOLDS}")

        train_gen, valid_gen = get_generators(df, image_dir, train_indices, valid_indices)
        binary_model = get_model()

        # Add early stopping callback
        early_stopping = EarlyStopping(monitor="val_loss", 
                                       patience=PATIENCE, restore_best_weights=True)

        # Add TensorBoard callback
        log_dir = "keras_logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = binary_model.fit(train_gen, validation_data=valid_gen, 
                                   epochs=NUM_EPOCHS, callbacks=[early_stopping, tensorboard_callback])

        # Evaluate the model on the validation set
        test_loss, test_accuracy = binary_model.evaluate(valid_gen)
        print(f"Test Accuracy for Fold {fold + 1}: {test_accuracy}")

        # Plot training history or perform any other analysis
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
