#%%
from keras.preprocessing.image import ImageDataGenerator

#%%

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(50, 50),
        batch_size=batch,
        class_mode='binary'
    )

    validation_generator = val_datagen.flow_from_directory(
        'data/validation',
        target_size=(50, 50),
        batch_size=batch,
        class_mode='binary')
