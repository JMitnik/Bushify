#%%
from data_generators import train_model, test_model
from network_models import make_alpha_model, make_transfer_model
transfer_model = make_transfer_model()
# transfer_model = train_model(transfer_model, 'transfer', 3)
# alpha_model = make_alpha_model()
(hist, transfer_model) = train_model(transfer_model, 'transfer', 30)
# print("Should be done training now!")
# from plots import plot_model_history
# alpha_model.load_weights('weights/alpha-run-2/weights-20-0.97.hdf5')
alpha_results = test_model(transfer_model)

#%%
print(alpha_results)
