# utils/preprocessing.py

import numpy as np
import cv2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def preprocess_images(images):
    images = [cv2.resize(img, (32, 32)) for img in images]
    images = np.array(images).astype("float32") / 255.0
    return images
