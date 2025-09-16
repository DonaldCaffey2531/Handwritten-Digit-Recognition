from tensorflow.keras.datasets import mnist
import os
from PIL import Image
import numpy as np

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Base directory to save images
base_dir = "../datasets/"
os.makedirs(base_dir, exist_ok=True)

# Create folders for each class
for i in range(10):
    os.makedirs(os.path.join(base_dir, "train", str(i)), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", str(i)), exist_ok=True)

# Save training images
for idx, (img_array, label) in enumerate(zip(x_train, y_train)):
    img = Image.fromarray(img_array)  # convert numpy array to image
    save_path = os.path.join(base_dir, "train", str(label), f"{idx}.png")
    img.save(save_path)

# Save test images
for idx, (img_array, label) in enumerate(zip(x_test, y_test)):
    img = Image.fromarray(img_array)
    save_path = os.path.join(base_dir, "test", str(label), f"{idx}.png")
    img.save(save_path)
