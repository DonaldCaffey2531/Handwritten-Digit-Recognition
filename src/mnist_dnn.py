# mnist_dnn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import io, json

# 1) Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# shape: (60000,28,28), (10000,28,28)

# 2) Preprocess
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
# Flatten for DNN
x_train_flat = x_train.reshape((-1, 28*28))
x_test_flat  = x_test.reshape((-1, 28*28))

# 3) Build DNN model (simple but effective on MNIST)
model = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 4) Callbacks
cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    callbacks.ModelCheckpoint('../models/mnist_dnn_best.keras', save_best_only=True)
]

# 5) Train
hist = model.fit(
    x_train_flat, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=128,
    callbacks=cb,
    verbose=2
)

# 6) Evaluate
test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

# 7) Plot training curves (optional)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hist.history['loss'], label='train loss'); plt.plot(hist.history['val_loss'], label='val loss'); plt.legend()
plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'], label='train acc'); plt.plot(hist.history['val_accuracy'], label='val acc'); plt.legend()
plt.show()

# 8) Confusion matrix + report
y_pred = np.argmax(model.predict(x_test_flat), axis=1)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix (shape):", cm.shape)

# 9) Save final model (SavedModel + h5)
model.save('../models/mnist_dnn.keras')        # h5 file
print("Models saved: mnist_dnn_saved/, mnist_dnn.h5")

# 10) helper: predict from a PIL image or array (for single-image inference)
def predict_pil_image(pil_img, model):
    # pil_img: grayscale or RGB, any size - we convert to 28x28 grayscale
    img = pil_img.convert('L').resize((28,28))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1, 28*28)
    # NOTE: depending on your input, you may need to invert colors if digits are black-on-white vs white-on-black
    probs = model.predict(arr)[0]
    return int(np.argmax(probs)), probs

# Example usage:
# from PIL import Image
# im = Image.open('my_digit.png')
# print(predict_pil_image(im, model))
