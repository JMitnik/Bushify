from flask import Flask, request, jsonify, make_response
from preprocessor import preprocessor
from keras import backend
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
model = None


@app.route("/")
def index():
    print(__name__)
    return "Hello"


@app.route("/upload_file", methods=['POST', 'PUT'])
def test_file_input():
    print("ADSASDas")
    file = request.files['image']
    (face_img, public_image, file_name) = preprocessor(file)
    face_array = np.expand_dims(img_to_array(face_img), axis=0)
    face_array = np.reshape(face_array, (1, 150, 150, 1))

    print("Face found!")
    print("Starting prediction!")
    prediction = model.predict(face_array)
    print("PREDICTION COMPLETE!!!")
    return jsonify(image_url=public_image, prediction=prediction.tolist())


def init_model():
    global model
    model = load_model('my_model.h5')
    print("Model has been loaded and is ready!")


if __name__ == "app":
  print(("* Loading Keras model and Flask starting server..."
         "please wait until server has fully started"))
  backend.clear_session()
  init_model()
  app.run(use_reloader=False, debug=True, )
