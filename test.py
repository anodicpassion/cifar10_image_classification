# test.py

import tensorflow as tf
import cv2
import numpy as np
from utils.config import *

# CIFAR-10 class names
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

model = tf.keras.models.load_model(MODEL_PATH)
img = load_and_preprocess_image("sample_image.jpg")  # Replace with your image path
prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]

print(f"Predicted class: {predicted_class}")
