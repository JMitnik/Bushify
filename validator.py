#%%
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#%%


def validate_model(model: Sequential):
    batch = 16

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(50, 50),
        batch_size=batch,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(50, 50),
        batch_size=batch,
        class_mode='binary')

    model.fit_generator(train_generator, steps_per_epoch=2000, epochs=10,
                        validation_data=validation_generator,validation_steps=800)
    model.save_weights('first_try.h5')

    return model
