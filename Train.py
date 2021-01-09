from cv2 import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
import os

# dev, For using GPU for training
# gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


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
    np.save("./imgs", imgs)
    np.save("./lbls", lbls)

    # Return The numpy arrays to caller
    return imgs, lbls


def Train():

    # Load images and labels
    # Expected shapes are
    # (n, 28, 28, 3) for images array
    # (n, 1) for labels array
    # Where n is the number of images, 28*28 is number of pixels
    # , and 3 is the number of color channels of an RGB image
    # uncomment next line
    # loaded_imgs, loaded_lbls = Load_data("./trainingSet")

    # dev
    loaded_imgs = np.load("./imgs.npy")
    loaded_lbls = np.load("./lbls.npy")
    # cv2.imshow("ho", loaded_imgs[0])
    # cv2.waitKey(0)

    # Divide into train/test sets with 98%/2% split
    numTrain = int(len(loaded_imgs) * 0.98)
    numTest = len(loaded_imgs) - numTrain

    x_train = loaded_imgs[:]
    y_train = loaded_lbls[:]

    x_test = loaded_imgs[:numTest]
    # y_test = loaded_lbls[:numTest]

    # Normalizing the pixels
    x_train = x_train / 255
    # x_test = x_test / 255

    # Use One-Hot encoding for labels
    y_train_one_hot = to_categorical(y_train, num_classes=10, dtype=np.uint8)
    # y_test_one_hot = to_categorical(y_test, num_classes=10, dtype=np.uint8)

    # Building the model
    model = keras.Sequential(
        [
            keras.layers.Conv2D(28, (4, 4), activation="relu", input_shape=(28, 28, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (4, 4), activation="relu"),
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

    # training the model
    model.fit(x_train, y_train_one_hot, batch_size=256, epochs=80, validation_split=0.2)

    # Saving the model
    model.save("./trained", overwrite=False, save_format="h5")


Train()