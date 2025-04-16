# evaluate.py

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from utils.preprocessing import load_dataset
from utils.config import *

model = tf.keras.models.load_model(MODEL_PATH)
_, x_val, _, y_val = load_dataset()

y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("Classification Report:\n")
print(classification_report(y_true, y_pred_classes))

print("Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred_classes))