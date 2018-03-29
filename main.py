#%%
from data_generators import train_model
from network_models import make_alpha_model, make_transfer_model
# transfer_model = make_transfer_model()
# transfer_model = train_model(transfer_model, 'transfer', 3)
alpha_model = make_alpha_model()
# (hist, alpha_model) = train_model(alpha_model, 'alpha', 30)
# print("Should be done training now!")
#%%
# from plots import plot_model_history
alpha_model.load_weights('weights/alpha-run-1/weights-end.hdf5')
