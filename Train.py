from cv2 import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


def Load_data(directory):
    imgs = []
    lbls = []
    for i in os.listdir(directory):
        for j in os.listdir(f"{directory}/{i}"):
            imgs.append(cv2.imread(f"{directory}/{i}/{j}"))
            lbls.append(int(i))
    imgs = np.array(imgs)
    lbls = np.array(lbls)
    lbls = np.reshape(lbls, (len(lbls), 1))
    # print(loaded_imgs, loaded_lbls)
    print("Images Array Shape: ", imgs.shape, " Labels Array Shape: ", lbls.shape)
    print("Images Array ndim: ", imgs.ndim, " Labels Array ndim: ", lbls.ndim)
    print("Images Array dtype: ", imgs.dtype, " Labels Array dtype: ", lbls.dtype)
    print("Images Array dtype Name: ", imgs.dtype.name, " Labels Array dtype Name: ", lbls.dtype.name)
    return imgs, lbls


loaded_imgs, loaded_lbls = Load_data("./trainingSet")