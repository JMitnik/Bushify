from flask import Flask, request, jsonify, make_response
from preprocessor import preprocessor
from keras import backend
from validator import validate_model
from keras.preprocessing.image import img_to_array, load_img
from build_model import model
import numpy as np

app = Flask(__name__)


@app.route("/")
def hello():
  dog = "test"
  return dog + dog

@app.route("/test", methods=['POST', 'PUT'])
def test_file_input():
    file = request.files['test']
    (face_img, file_name) = preprocessor(file)
    face_array = np.expand_dims(img_to_array(face_img), axis=0)

    print("Starting prediction!")
    prediction = model.predict(face_array)
    return make_response(jsonify(image_url=file_name, bush=prediction))

if __name__ == "__main__":
  print("Prevalidation")
  model = validate_model(model, 10)
  print("Running app")
  app.run()
