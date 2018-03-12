#%%
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

#%%
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augmentImagesInPath(images, path):
    for image in images:
        img = load_img(path+'/'+image)
        augmentImage(img)

def augmentImage(image):
    flattened_img = img_to_array(image)
    flattened_img = flattened_img.reshape((1,) + flattened_img.shape)
    generateAugments(flattened_img, 1, 20)

def generateAugments(image, batch_size, number_augments):
    i = 0

    for batch in datagen.flow(image, batch_size=batch_size, save_to_dir="data/keras-test", save_prefix='test', save_format="jpeg"):
        i +=1

        if i > number_augments:
            break
