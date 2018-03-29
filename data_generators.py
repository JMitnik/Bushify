from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import pandas as pd
import shutil
import numpy as np
import glob
import cv2
import math
import pickle
import datetime
from keras.utils import np_utils
from FaceModel import FaceModel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback, TensorBoard
from time import time

outputFolder = './output-mnist'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath = outputFolder+"/weights-{epoch:02d}-{val_acc:.2f}.hdf5"

endfilepath = outputFolder+"/weights-end.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=False, save_weights_only=False,
                             mode='auto', period=10)
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5,
                        verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

callbacks_list = [checkpoint, earlystop, tensorboard]

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    return img

def encodeLabels(labels):
    return LabelEncoder().fit_transform(labels)

def load_train():
    X_train = []
    y_train = []
    classes = ['0-not_bush', '1-bush']
    print('Read train images')
    for c in classes:
        print('Load folder class {}'.format(c))
        path = os.path.join('data', 'faces',
                            'train', c, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
            y_train.append(c)

    return X_train, y_train

def load_test():
    x_test = []
    y_test = []
    classes = ['0-not_bush', '1-bush']
    print('Read test images')
    for c in classes:
        print('Load folder class {}'.format(c))
        path = os.path.join('data', 'faces',
                            'test', c, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            x_test.append(img)
            y_test.append(c)

    return (x_test, y_test)


def make_default_image_generators():
    train_image_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_image_gen = ImageDataGenerator(rescale=1./255)

    return (train_image_gen, val_image_gen)

# def make_data_generators(source_glob=None, dimension_tuple=(150,150), batch_size=16):
#     (train_img_gen, val_img_gen) = make_default_image_generators()

#     try:
#         train_data_gen = train_img_gen.flow_from_directory('data'+source_glob+'training', target_size=dimension_tuple, batch_size=batch_size, class_mode='binary')
#         val_data_gen = val_img_gen.flow_from_directory('data'+source_glob+'validation', target_size=dimension_tuple, batch_size=batch_size, class_mode='binary')
#         return (train_data_gen, val_data_gen)
#     except Exception as e:
#         print(e)

def format_data(train_x, train_y):
    train_x = np.array(train_x, dtype=np.uint8)
    train_y = np.array(train_y)
    train_x = train_x.reshape(train_x.shape[0], 150, 150, 1)
    train_y = encodeLabels(train_y)

    return (train_x, train_y)

# def cross_validate_test(make_model_callback, folds=10, epochs=10, batch_size=16):
#     (train_image_gen, val_image_gen) = make_default_image_generators()

#     (X, Y) = load_train()
#     (X, Y) = format_data(X, Y)

#     kfolds = StratifiedKFold(folds)
#     eval_result = []

#     for (train_idx, val_idx) in kfolds.split(X, Y):
#         model = FaceModel(make_model_callback)
#         x_train, x_valid = X[train_idx], X[val_idx]
#         y_train, y_valid = Y[train_idx], Y[val_idx]

#         train_data = (x_train, y_train)
#         valid_data = (x_valid, y_valid)
#         model.validation_train(train_data, valid_data, (train_image_gen, val_image_gen), epochs, batch_size)
#         eval_result.append(model)

def get_train_and_valid_data(valid_size=0.2):
    (X, Y) = load_train()
    (X, Y) = format_data(X, Y)

    (x_train, x_valid, y_train, y_valid) = train_test_split(X, Y, test_size=valid_size)

    train_data = (x_train, y_train)
    valid_data = (x_valid, y_valid)
    return (train_data, valid_data)

def train_model(model, model_name="name", epochs=50, batch_size=16):
    (train_image_gen, val_image_gen) = make_default_image_generators()
    (train_data, valid_data) = get_train_and_valid_data()
    history = model.fit_generator(train_image_gen.flow(train_data[0], train_data[1], batch_size),
        callbacks=callbacks_list, epochs=epochs, validation_data=val_image_gen.flow(valid_data[0], valid_data[1]))
    print("Done with the fitting!")
    model.save(endfilepath)
    print("Saved last epoch!")
    return history, model


def format_test_data(x, y):
    test_x = np.array(x, dtype=np.uint8)
    test_y = np.array(y)
    test_x = test_x.reshape(test_x.shape[0], 150, 150, 1)
    test_y = encodeLabels(test_y)

    return (test_x, test_y)

def confusion_matrix(model, X, Y):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for i in range(X.shape[0]):
        test_run = np.expand_dims(X[i], axis=0)
        prediction = np.round(model.predict(test_run))
        label = Y[i]

        if (label == 1):
            if prediction[0][0] == label:
                TP += 1

            else:
                FN += 1
        else:
            if prediction[0][0] == label:
                TN += 1
            else:
                FP += 1

    return (TP, FN, FP, TN)

def test_model(model):
    (X, Y) = load_test()
    (X, Y) = format_test_data(X, Y)

    (TP, FN, FP, TN) = confusion_matrix(model, X, Y)
    recall = (TP / (TP + FN))
    precision = (TP / (TP + FP))
    print(Y.shape)
    accuracy = ((TP + TN) / Y.shape[0])

    results = {
        'TP': TP,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy
    }

    return results
