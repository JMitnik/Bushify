from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

# Default Keras Neural Network with 3 conv2d-block
def make_alpha_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 1), dim_ordering="tf"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])

    return model

def make_transfer_model():
    transfer_model = VGG16(include_top=False, weights='imagenet', input_shape=(50, 50, 3))

    full_connected_bush_block = Sequential()

    full_connected_bush_block.add(
        Flatten(input_shape=transfer_model.output_shape[1:]))
    full_connected_bush_block.add(Dense(256, activation='relu'))
    full_connected_bush_block.add(Dropout(0.5))
    full_connected_bush_block.add(Dense(1, activation='sigmoid'))

    end_model = Sequential()

    for i in transfer_model.layers:
        end_model.add(i)

    end_model.add(full_connected_bush_block)

    for i in end_model.layers[:19]:
        i.trainable = False

    end_model.compile(loss='binary_crossentropy',
                      optimizer=Adam(), metrics=['accuracy'])

    return end_model
