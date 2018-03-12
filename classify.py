#%%
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#%%
model = Sequential()

# So, 32 filters, each with size (3,3)
model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3), dim_ordering="tf"))

# Gets rid of negative value
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#%%
batch = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(50,50),
    batch_size=batch,
    class_mode='binary'
)

model.fit_generator(train_generator, steps_per_epoch=2000, epochs=10)


model.save_weights('test.h5')
#%%
