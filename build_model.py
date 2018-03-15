from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


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
print(model)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])

def train_model(model, epochs=50, validation_gen=None, validation_steps=800):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(50, 50),
        batch_size=16,
        class_mode='binary'
    )

    if validation_gen:
        model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=epochs,
            validation_data=validation_gen,
            validation_steps=800)
    else:
        model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=epochs
        )

    return model
