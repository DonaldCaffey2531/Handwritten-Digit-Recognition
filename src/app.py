# app.py
from flask import Flask, request, jsonify
import base64, io
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('../models/mnist_dnn.keras')

def prepare_image_from_base64(b64str):
    raw = base64.b64decode(b64str)
    img = Image.open(io.BytesIO(raw)).convert('L').resize((28,28))
    arr = np.array(img).astype('float32')/255.0
    arr = arr.reshape(1, 28*28)
    return arr

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    b64 = data.get('image')
    if not b64:
        return jsonify({'error': 'no image provided'}), 400
    arr = prepare_image_from_base64(b64)
    probs = model.predict(arr)[0]
    pred = int(np.argmax(probs))
    return jsonify({'pred': pred, 'probs': probs.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
