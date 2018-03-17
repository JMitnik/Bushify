#%%
import os
import matplotlib.pyplot as plt
import cv2
import augment

abs_path = os.path.dirname(os.path.abspath("__file__"))
CASCADE_FILE_SRC = abs_path+"/model/haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_FILE_SRC)

def preProcessImage(img):
    cv2_image = cv2.imread(img)
    grayscaled_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    try:
        face_coordinates = detectFace(grayscaled_image)
        (x, y, w, h) = face_coordinates
    except IndexError:
        raise

    face = cv2_image[y:y+w, x:x+h]
    resized_face = cv2.resize(face, (150, 150))

    return resized_face

def detectFace(img):
    faces = FACE_CASCADE.detectMultiScale(img, 1.2, 5)

    for (x, y, w, h) in faces:
        box = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return faces[0]

def writeImage(img, output_path, filename):
    cv2.imwrite(abs_path+'/data/faces/'+output_path+'/'+filename, img)

def preProcessDataSet(files, source_path, output_path):
    for file in files:
        try:
            pp_image = preProcessImage(source_path+'/'+file)
            writeImage(pp_image, output_path, file)
        except Exception as e:
            print("No image recognized")
    return

start_folder = abs_path+'/data/normal'

def recursiveWalk(start_folder):
    for current, dirs, files in os.walk(start_folder):
        if any('.jpg' in i for i in files):
            preProcessDataSet(files, current ,current[(len(start_folder) + 1):])

recursiveWalk(start_folder)
