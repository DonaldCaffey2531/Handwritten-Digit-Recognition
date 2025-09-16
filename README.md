# Handwritten-Digit-Recognition

# Handwritten Digit Recognition Using DNN Project Overview

This project implements a Deep Neural Network (DNN) to recognize handwritten digits (0–9) using the MNIST dataset. The project supports both training from scratch and predicting new images. It also includes options to load MNIST as images on your local machine.

# Features

Train a fully connected DNN on MNIST digits.

Save and load trained models (.h5 or TensorFlow SavedModel format).

Predict single images from local files.

Support for local datasets (custom digits saved as images).

Optional deployment via Flask API for real-time inference.

# Requirements

Python 3.10+

TensorFlow 2.x

NumPy

Pillow

Matplotlib (optional, for plotting training curves)

Flask (optional, for API deployment)

Install dependencies:

pip install tensorflow numpy pillow matplotlib flask

# Usage
1. Training the DNN
    python mnist_dnn.py

    Loads MNIST from Keras or local image folders.

    Trains a DNN with early stopping and saves the best model.

    Produces training/validation accuracy and loss plots.

2. Predict a single image
    from mnist_dnn import predict_pil_image
    from PIL import Image
    from tensorflow.keras.models import load_model

    model = load_model('mnist_dnn.h5')
    img = Image.open('my_digit.png')
    predicted_class, probabilities = predict_pil_image(img, model)
    print(f"Predicted digit: {predicted_class}")

3. Run Flask API (optional)
    python app.py


    POST request to /predict with JSON:

    {
    "image": "<base64_encoded_image>"
    }

    Returns predicted digit and probabilities.

# Local MNIST Images

    You can convert the MNIST dataset to local image files for visualization or custom training:

    python save_mnist_as_images.py


    Creates train/0-9 and test/0-9 folders.

    Each image is saved as a .png file.

    Model Performance

    Typical accuracy with the DNN: ~97% on MNIST test set.

    Can be improved using CNNs, data augmentation, or ensembles.

# References

MNIST Dataset

TensorFlow Keras Documentation

PyTorch Documentation

# License

This project is open-source and free to use for educational and research purposes.