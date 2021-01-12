import numpy as np
import tensorflow as tf
from tensorflow import keras
from cv2 import cv2
import sys
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


# For colored priniting
def printInColor(str):
    print(f"\033[92m{str}\033[0m")


# Defining out of try/catch scope to be globally accessible
model = None


try:

    # Try to load pretrained model, if not found, train the model and save it.
    model = keras.models.load_model(f"{rootDir}/trainedmodel.h5")

    # Printing the model summary
    # model.summary(print_fn=printInColor)

    # Printing Trained Parameters, w and b
    printInColor(f"Trained Parameters: {model.get_weights()}")

    # Taking the value to predict as input from the user
    imPath = input(f"\033[92mEnter The Path to the image you want to predict\n\033[0m")

    x_to_predict = np.array([cv2.imread(imPath)])
    print(x_to_predict.shape)

    # Using the model to get predictions
    prediction = model.predict(x_to_predict)
    printInColor(f"Predicted Class: {[np.where(r==1)[0][0] for r in prediction]}")

except:
    printInColor(f"The trained model '{rootDir}/trainedmodel.h5' was not found or some other error occured\n")
    print(sys.exc_info())