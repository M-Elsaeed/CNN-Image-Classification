import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from cv2 import cv2
import sys

# dev, For using GPU for training
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# For colored priniting
def printInColor(str):
    print(f"\033[92m{str}\033[0m")


# Defining out of try/catch scope to be globally accessible
model = None


try:

    # Try to load pretrained model, if not found, train the model and save it.
    model = keras.models.load_model("./5_5_shuffled.h5")

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
    printInColor(f"Predicted Class: {prediction}")

except:
    printInColor("The trained model './trainedmodel.h5' was not found or someother error occured\n")
    print(sys.exc_info())