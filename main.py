#%%
from data_generators import train_model
from network_models import make_alpha_model, make_transfer_model

transfer_model = make_transfer_model()
transfer_model = train_model(transfer_model, 'transfer')
alpha_model = make_alpha_model()
alpha_model = train_model(alpha_model, 'alpha')

# Vergelijk alpha_model en #transfer_model
