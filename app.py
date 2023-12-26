from flask import Flask, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def preprocess_image(img, img_height=299, img_width=299):
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0
    img = img.astype(np.float32)
    return img

def predict_fingerprint_tflite(interpreter, preprocessed_img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, preprocessed_img)

    interpreter.invoke()

    output_index = output_details[0]['index']
    output = interpreter.get_tensor(output_index)

    predicted_class = np.argmax(output)

    return predicted_class, output

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    try:
        image_file = request.files['image'].read()
        nparr = np.frombuffer(image_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        interpreter = tf.lite.Interpreter(model_path='model.tflite')
        interpreter.allocate_tensors()

        preprocessed_img = preprocess_image(img)

        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

        predicted_class, predictions = predict_fingerprint_tflite(interpreter, preprocessed_img)

        return jsonify({
            'predicted_class': int(predicted_class),
            'predictions': predictions.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
