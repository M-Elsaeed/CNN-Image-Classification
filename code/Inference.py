import sys
import platform
import numpy as np
from cv2 import cv2
import silence_tensorflow.auto  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow import keras

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
def printInColor(inStr):
    if platform.system() == "Windows":
        print(inStr)
    else:
        print(f"\033[92m{inStr}\033[0m")

# Loads model and runs prediciton on the image with the passed path
def predict(imPath):
    # Defining out of try/catch scope to be globally accessible
    model = None

    try:

        # Try to load pretrained model, if not found, train the model and save it.
        model = keras.models.load_model(f"{rootDir}/trainedmodel.h5")

        # Printing the model summary
        # model.summary(print_fn=printInColor)

        # Printing Model's Summary
        model.summary(print_fn=printInColor)

        #  Print the path of the image on which inference will be run.
        printInColor(f"Running infernece on image located at{imPath}")

        x_to_predict = np.array([cv2.imread(imPath)])
        print(x_to_predict.shape)

        # Using the model to get predictions
        prediction = model.predict(x_to_predict)
        print(prediction)
        printInColor(f"Predicted Class: {[np.where(r==1)[0][0] for r in prediction]}")

    except:
        printInColor(
            f"The trained model '{rootDir}/trainedmodel.h5' was not found or some other error occured\n"
        )
        print(sys.exc_info())


# Main Function: Loads the data, then runs the prediciton on either the path provided in system arguments
# or path taken from user prompt if sys argument is not provided
if __name__ == "__main__":
    # Taking the image's path to predict as argument from CLI (for docker)
    # OR Taking the image's path to predict as input from the user
    _imPath = None
    if len(sys.argv) > 1:
        _imPath = sys.argv[1]
    else:
        printInColor("Enter The Path to the image you want to predict")
        _imPath = input()
    predict(_imPath)