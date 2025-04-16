# 🧠 CIFAR-10 Image Classification with TensorFlow & OpenCV

This project implements a high-accuracy image classifier for the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), using **TensorFlow** and **OpenCV**. It trains a Convolutional Neural Network (CNN) to recognize 10 object categories from tiny 32x32 color images.


---

## 💿 Dataset

The dataset contains **60,000 color images** across **10 classes**, split into 50,000 training and 10,000 test images. Each image is `32x32` pixels with RGB channels.

**Classes:**
- `airplane`
- `automobile`
- `bird`
- `cat`
- `deer`
- `dog`
- `frog`
- `horse`
- `ship`
- `truck`

## 📁Folder Structure
```
.
├── LICENSE
├── README.md
├── data
│   ├── README.md
│   └── file_structure
├── evaluate.py
├── models
│   └── model.h5
├── requirements.txt
├── test.py
├── train.py
└── utils
    ├── __pycache__
    │   ├── config.cpython-312.pyc
    │   └── preprocessing.cpython-312.pyc
    ├── config.py
    └── preprocessing.py

5 directories, 13 files
```

## 🛠️ Features

- Custom OpenCV-based image loader and preprocessor
- CNN model using TensorFlow/Keras
- Easily extendable for transfer learning
- Modular file structure with separate training, evaluation, and test scripts
- Confusion matrix and classification report
- Predict on custom images with `test.py`

---

## 🚀 Setup

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

### Download the data

```html
https://www.kaggle.com/c/cifar-10/
```

### Training 

```shell
python3 trian.py
```
This will train the model and store it into `model/model.h5`.

### Evaluation

```shell
python3 evaluate.py
```

### Testing

```shell
python3 test.py
```

## 📜 License

This project is licensed under the GNU General Public License v3.0.