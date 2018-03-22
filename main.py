#%%
from network_models import make_alpha_model, make_transfer_model
from data_generators import cross_validate_test

# transfer_model = FaceModel(make_transfer_model)
cross_validate_test(make_alpha_model)
