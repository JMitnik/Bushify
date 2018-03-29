import cv2
import os
import io
import numpy as np
from werkzeug.utils import secure_filename
abs_path = os.path.dirname(os.path.abspath("__file__"))

UPLOAD_FOLDER = 'app/public/uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
CASCADE_FILE_SRC = abs_path+"/model/haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_FILE_SRC)

def preprocessor(file):
    filename = file.filename.replace(" ", "-")
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        box = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    (x, y, w, h) = faces[0]

    croppedFace = gray[y:y+w, x:x+h]

    testFace = cv2.resize(croppedFace, (150, 150))
    cv2.imwrite(UPLOAD_FOLDER+filename, testFace)
    cv2.imwrite(UPLOAD_FOLDER+"rect_"+filename, img)

    # file.save(os.path.join(UPLOAD_FOLDER, filename))
    return (testFace, 'uploads/rect_'+filename, 'uploads/'+filename)
