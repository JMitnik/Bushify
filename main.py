#%%
from network_models import make_alpha_model, make_transfer_model
from data_generators import
from FaceModel import FaceModel

transfer_model = FaceModel(make_transfer_model)
alpha_model = FaceModel(make_alpha_model)
