import os
import platform
import numpy as np
from cv2 import cv2
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical


rootDir = "/home"

# If running on Windows (GPU used, and using relative paths instead of absolute)
if platform.system() == "Windows":
    # dev, For using GPU for training
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Use relative path if Windows is used (or not using docker)
    rootDir = "./code"


# For colored priniting
def printInColor(str):
    print(f"\033[92m{str}\033[0m")


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
            # Image resizing layer.Resize the batched image input to target height and width.
            # The input should be a 4-D tensor
            keras.layers.experimental.preprocessing.Resizing(
                28, 28, interpolation="bilinear"
            ),
            # Output Shape: (n, 28, 28, 3), where n is the number of images fed to the model
            
            
            # 2D Convulutional Layer with 25 filters, and kernel size(3, 3), and relu activation function.
            keras.layers.Conv2D(
                25, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 3)
            ),
            # Calculation of output shape:
            # W is the input size(width or height since they are the same in our case) = 28
            # K is the Kernel size = 3
            # P is the padding = 0 (Valid Padding)
            # S is the stride = 1
            # [(W−K+2P)/S]+1
            # = (28-3) + 1 = 26
            # 26 * 26 features with 25 channels for each features because we had 25 filters
            # Output Shape: (n, 26, 26, 25), where n is the number of images fed to the model
            
            
            # Maxpooling layer to reduce size and increase robustness of model
            keras.layers.MaxPool2D(),
            # Calculation of output shape:
            # W is the input size(width or height since they are the same in our case) = 26
            # K is the Kernel size = 2
            # P is the padding = 0 (Valid Padding)
            # S is the stride = 2
            # [(W−K+2P)/S]+1
            # = [(26-2)/2] + 1 = 13
            # 13 * 13 features with 25 channels for each features because we had 25 filters from the previos layer
            # Output Shape: (n, 13, 13, 25), where n is the number of images fed to the model
            
            
            # Flattenning layer converts all features into a 1-D vector
            keras.layers.Flatten(),
            # Calculation of output shape:
            # 13 * 13 * 25 = 4225
            # Output Shape: (n, 4225), where n is the number of images fed to the model
            
            
            # Dense(Fully connected) layer with 250 units(neurons) and relu activaiton function
            keras.layers.Dense(250, activation="relu"),
            # Calculation of output shape: Same as number of units(neurons)
            # Output Shape: (n, 250), where n is the number of images fed to the model

            # Dense(Fully connected) layer with 10 units(neurons, same numebr of classes)
            # and softmax Activation function. It represents a one-hot encoding of the number of classes
            keras.layers.Dense(10, activation="softmax")
            # Calculation of output shape: Same as number of units(neurons)
            # Output Shape: (n, 10), where n is the number of images fed to the model   
        ]
    )

    adamOptimizer = keras.optimizers.Adam(learning_rate=0.001)
    # Model compilation using ADAM optimizer
    # Categorical Crossentropy as loss funciton
    # and accuracy as the metric for training
    model.compile(
        loss="categorical_crossentropy", optimizer=adamOptimizer, metrics=["accuracy"]
    )

    # BONUS1, tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{rootDir}/logs")

    # training the model
    model.fit(
        x_train,
        y_train_one_hot,
        # To fit in memmory easily
        batch_size=128,
        # Through Trials, 10 epocs yielded acceptable accuracy and loss results
        epochs=10,
        # Using 98% of the data for training
        validation_split=0.02,
        # for BONUS1 Tensorboard
        callbacks=[tensorboard_callback]
    )

    # Printing the model summary
    model.summary(print_fn=printInColor)

    # Predict some random images and show the correct and predicted labels.
    # Will not work on docker.
    showIfWindows(model, x_train, y_train)

    # Saving the model
    model.save(f"{rootDir}/trainedmodel.h5", save_format="h5")


# Shuffles the passed arrays while maintaining correspondance
# uses numpy permutation generator
def shuffle(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


# Picks a random sample from images and displays their Labels
# Along side their predicted labels using the passed model for prediction
# Only works on Windows/full-blown ubuntu, not on docker
def showIfWindows(model, images, orig_labels):
    if platform.system() == "Windows":
        # 3 images shown randomly
        randomIndices = np.random.choice(len(y_train), 5)
        for i in randomIndices:
            rand_im = images[i]
            rand_im_lbl_orig = orig_labels[i]
            rand_im_lbl_pred = model.predict(np.array([rand_im* 255]))
            rand_im = cv2.copyMakeBorder(
                rand_im, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0
            )
            # print(f"Y: {rand_im_lbl_orig}, Y_pred: {rand_im_lbl_pred}")
            cv2.putText(
                rand_im,
                f"Y: {rand_im_lbl_orig}, Y_pred: {[np.where(r==1)[0][0] for r in rand_im_lbl_pred]}",
                (5, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Sample", rand_im)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        printInColor("Showing Images only works on OS with GUI")


if __name__ == "__main__":
    # Load images and labels
    # Expected shapes are
    # (n, 28, 28, 3) for images array
    # (n, 1) for labels array
    # Where n is the number of images, 28*28 is number of pixels
    # , and 3 is the number of color channels of an RGB image
    # The model is adapted to any image size by modifying its imput layer
    x_train = None
    y_train = None
    # An attempt is made to load from pre-read NPY files
    try:
        # dev
        x_train = np.load(f"{rootDir}/imgs.npy")
        y_train = np.load(f"{rootDir}/lbls.npy")

        print("DATA LOADED FROM PREREAD NPY FILES")
    # If no NPY files are found, use the load function
    except:
        x_train, y_train = Load_data(f"{rootDir}/trainingSet")

        print("DATA LOADED FROM FILE SYSTEM")

    Train(x_train, y_train)