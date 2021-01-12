from cv2 import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import os
import platform


rootDir = "/home/16p8160"

# If running on Windows (GPU used, and using relative paths instead of absolute)
if platform.system() == "Windows":
    # dev, For using GPU for training
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Use relative path if Windows is used (or not using docker)
    rootDir = "./project"


# Takes directory to load images from then saves the read images and labels
# Inf f"{rootDir}/imgs.npy" andf "{rootDir}/lbls.npy" respectively
# As a binary file in NumPy .npy format
#
# expected directory structure is
# |root
# | | Lbl_1 | Img_1.jpg
# | | Lbl_1 | Img_2.jpg
# | | Lbl_1 |     .
# | | Lbl_1 |     .
# | | Lbl_1 |     .
# | | Lbl_1 | Img_X.jpg
# | | Lbl_2 | Img_1.jpg
# | | Lbl_2 | Img_2.jpg
# | | Lbl_2 |     .
# | | Lbl_2 |     .
# | | Lbl_2 |     .
# | | Lbl_2 | Img_X.jpg
# .
# .
# .
# | | Lbl_N | Img_1.jpg
# | | Lbl_N | Img_2.jpg
# | | Lbl_N |     .
# | | Lbl_N |     .
# | | Lbl_N |     .
# | | Lbl_N | Img_X.jpg
def Load_data(directory):

    # Initializing Empty Arrays
    imgs = []
    lbls = []

    # For each folder in the directory where the folder name is the label
    for i in os.listdir(directory):
        # For each file (image) in each folder in the directory, append the read image and
        # its respective label to the respective arrays
        for j in os.listdir(f"{directory}/{i}"):
            imgs.append(cv2.imread(f"{directory}/{i}/{j}"))
            lbls.append(int(i))

    # Convert the arrays into numpy arrays with unsigned int data types
    imgs = np.array(imgs, dtype=np.uint8)
    lbls = np.array(lbls, dtype=np.uint8)
    lbls = np.reshape(lbls, (len(lbls), 1))

    # Print Info about the read data
    print("Images Array Shape: ", imgs.shape, " Labels Array Shape: ", lbls.shape)
    print("Images Array ndim: ", imgs.ndim, " Labels Array ndim: ", lbls.ndim)
    print("Images Array dtype: ", imgs.dtype, " Labels Array dtype: ", lbls.dtype)
    print(
        "Images Array dtype Name: ",
        imgs.dtype.name,
        " Labels Array dtype Name: ",
        lbls.dtype.name,
    )

    # dev
    np.save(f"{rootDir}/imgs", imgs)
    np.save(f"{rootDir}/lbls", lbls)

    # Return The numpy arrays to caller
    return imgs, lbls


def Train(x_train=None, y_train=None):

    # Shuffling the data
    x_train, y_train = shuffle(x_train, y_train)

    # Normalizing the pixels
    x_train = x_train / 255

    # Use One-Hot encoding for labels
    y_train_one_hot = to_categorical(y_train, num_classes=10, dtype=np.uint8)

    # Building the model
    model = keras.Sequential(
        [
            keras.layers.experimental.preprocessing.Resizing(
                28, 28, interpolation="bilinear"
            ),
            keras.layers.Conv2D(28, (5, 5), activation="relu", input_shape=(28, 28, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (5, 5), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(1000, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(500, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(250, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Model compilation using ADAM optimizer
    # Categorical Crossentropy as loss funciton
    # and accuracy as the metric for training
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # BONUS1, tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{rootDir}/logs")

    # training the model
    model.fit(
        x_train,
        y_train_one_hot,
        batch_size=256,
        epochs=80,
        validation_split=0.02,
        callbacks=[tensorboard_callback],
    )

    # Printing the model summary
    model.summary()

    # Saving the model
    model.save(f"{rootDir}/trainedmodel.h5", overwrite=False, save_format="h5")


def shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == "__main__":
    # Load images and labels
    # Expected shapes are
    # (n, 28, 28, 3) for images array
    # (n, 1) for labels array
    # Where n is the number of images, 28*28 is number of pixels
    # , and 3 is the number of color channels of an RGB image
    x_train = None
    y_train = None
    try:
        # dev
        x_train = np.load(f"{rootDir}/imgs.npy")
        y_train = np.load(f"{rootDir}/lbls.npy")

        print("DATA LOADED FROM PREREAD NPY FILES")
    except:
        print("HEREEEEEE", os.listdir("./"))
        x_train, y_train = Load_data(f"{rootDir}/trainingSet")

        print("DATA LOADED FROM FILE SYSTEM")

    Train(x_train, y_train)