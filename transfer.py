from keras.applications import VGG16
from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, MaxPooling2D, Input
from keras.models import Sequential
from keras.optimizers import Adam

def get_transfer_model():
    transfer_model = VGG16(include_top=False, weights='imagenet', input_shape=(50,50, 3))

    full_connected_bush_block = Sequential()

    full_connected_bush_block.add(Flatten(input_shape=(50, 50, 3)))
    full_connected_bush_block.add(Dense(256, activation='relu'))
    full_connected_bush_block.add(Dropout(0.5))
    full_connected_bush_block.add(Dense(1, activation='sigmoid'))

    end_model = Sequential()

    for i in transfer_model.layers:
        end_model.add(i)

    end_model.add(full_connected_bush_block)

    for i in end_model.layers[:25]:
        i.trainable = False

    end_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return end_model
