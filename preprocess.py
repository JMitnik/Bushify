#%%
import os
import matplotlib.pyplot as plt
import cv2
import augment

abs_path = os.path.dirname(os.path.abspath("__file__"))
bush_path = abs_path+'/data/all_og/'
not_bush_path = abs_path+'/data/not_bush/'
pp_bush = abs_path+'/data/pp_bush/'
pp_not_bush = abs_path+'/data/pp_not_bush/'

CASCADE_FILE_SRC = abs_path+"/model/haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_FILE_SRC)

def preProcessImage(img, filename):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        face = detectFace(gray)
    except IndexError:
        return

    resizedFace = cv2.resize(face, (50,50))
    print(resizedFace)
    writeImage(resizedFace, filename)

def detectFace(img):
    faces = FACE_CASCADE.detectMultiScale(img, 1.2, 5)

    for (x, y, w, h) in faces:
        box = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    (x, y, w, h) = faces[0]

    croppedFace = img[y:y+w, x:x+h]

    return croppedFace

def writeImage(img, filename):
    fileName = 'pp_'+filename
    cv2.imwrite(abs_path+'/data/pp_notbush/'+fileName, img)


def preProcessDataSet(src, filenames):
    for name in filenames:
        preProcessImage(src+name, name)

    augment.augmentImagesInPath(filenames, src)
    return

preProcessDataSet(not_bush_path, os.listdir(not_bush_path))

#%%
