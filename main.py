#%%
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
# from build_model import model, train_model
from transfer import get_transfer_model
from validator import validate_model

transfer_model = get_transfer_model()
# model = validate_model(model, 20)
model = validate_model(transfer_model, 10)

#%%
print("test")
x = np.expand_dims(img_to_array(load_img('bush.jpg')), axis=0)
bushPred = model.predict(x)
print(bushPred)
# not_bush_pred = model.predict(img_to_array(load_img('test_not_bush.jpg')))
