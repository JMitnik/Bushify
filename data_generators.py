from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold

def make_default_image_generators():
    train_image_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_image_gen = ImageDataGenerator(rescale=1./255)

    return (train_image_gen, val_image_gen)

def make_data_generators(img_generator_tuple, source_glob=None, dimension_tuple=(50,50), batch_size=16):
    (train_img_gen, val_img_gen) = make_default_image_generators()

    if source_glob != None:
        source_glob = '/' + source_glob + '/'
    else:
        source_glob = '/'

    try:
        train_data_gen = train_img_gen.flow_from_directory('data'+source_glob+'training', target_size=dimension_tuple, batch_size=batch_size, class_mode='binary')
        val_data_gen = val_img_gen.flow_from_directory('data'+source_glob+'validation', target_size=dimension_tuple, batch_size=batch_size, class_mode='binary')
        return (train_data_gen, val_data_gen)
    except Exception as e:
        print(e)


# def cross_folds(k=10):


# sm_gray_data = make_data_generators(make_default_image_generators, 'gray', (50, 50))
# md_gray_data = make_data_generators(make_default_image_generators, 'gray', (100, 100))
# lg_gray_data = make_data_generators(make_default_image_generators, 'gray', (150, 150))

# sm_color_data = make_data_generators(make_default_image_generators, dimension_tuple=(50, 50))
# md_color_data = make_data_generators(make_default_image_generators, dimension_tuple=(100, 100))
# lg_color_data = make_data_generators(make_default_image_generators, dimension_tuple=(150, 150))
