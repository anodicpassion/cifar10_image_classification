# utils/preprocessing.py

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils.config import IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES

def load_dataset():
    labels_path = "data/cifar-10/trainLabels.csv"
    train_dir = "data/cifar-10/train/"
    test_dir = "data/cifar-10/test/"

    df = pd.read_csv(labels_path)

    label_map = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['label_idx'] = df['label'].map(label_map)

    X = []
    y = []

    for _, row in df.iterrows():
        image_id = row['id']
        label_idx = row['label_idx']
        img_path = os.path.join(train_dir, f"{image_id}.png")

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype('float32') / 255.0

            X.append(img)
            y.append(label_idx)

    X = np.array(X)
    y = to_categorical(y, NUM_CLASSES)

    return train_test_split(X, y, test_size=0.2, random_state=42)

def load_test_images():
    test_dir = "data/cifar-10/test/"
    test_images = []
    image_ids = []

    for fname in sorted(os.listdir(test_dir)):
        if fname.endswith(".png"):
            img_path = os.path.join(test_dir, fname)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype("float32") / 255.0
            test_images.append(img)
            image_ids.append(os.path.splitext(fname)[0])

    return np.array(test_images), image_ids