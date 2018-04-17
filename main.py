#%%
from data_generators import train_model, test_model
from network_models import make_alpha_model, make_transfer_model
import matplotlib.pyplot as plt
alpha_model = make_alpha_model()
transfer_model = make_transfer_model()
(hist1, transfer_model) = train_model(transfer_model, 'transfer', 20)
(hist2, alpha_model) = train_model(alpha_model, 'transfer', 20)
print("Should be done training now!")

#%%
# from plots import plot_model_acc_history, plot_model_prec_rec_history
import numpy as np


def plot_model_prec_rec_history(models):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].set_title('Model Precision')
    axs[0].set_ylabel('Precision')
    axs[0].set_xlabel('Epoch')
    axs[1].set_title('Model Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_xlabel('Epoch')

    for model_history in models:
        axs[0].plot(range(1, len(model_history.history['precision'])+1),
                    model_history.history['precision'])
        axs[0].plot(range(1, len(model_history.history['val_precision'])+1),
                    model_history.history['val_precision'], linestyle='--')
        axs[0].set_xticks(np.arange(
            1, len(model_history.history['precision'])+1), len(model_history.history['precision'])/10)
        axs[0].legend(['train_alpha', 'val_alpha',
                       'train_transfer', 'val_transfer'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1, len(model_history.history['recall'])+1),
                    model_history.history['recall'])
        axs[1].plot(range(1, len(model_history.history['val_recall'])+1),
                    model_history.history['val_recall'], linestyle='--')
        axs[1].set_xticks(np.arange(
            1, len(model_history.history['recall'])+1), len(model_history.history['recall'])/10)
        axs[1].legend(['train_alpha', 'val_alpha',
                       'train_transfer', 'val_transfer'], loc='best')
    plt.show()



def plot_models_acc_history(models):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')

    for model_history in models:
        axs[0].plot(range(1, len(model_history.history['acc'])+1),
                    model_history.history['acc'])
        axs[0].plot(range(1, len(model_history.history['val_acc'])+1),
                    model_history.history['val_acc'], linestyle='--')
        axs[0].set_xticks(np.arange(
            1, len(model_history.history['acc'])+1), len(model_history.history['acc'])/10)
        axs[0].legend(['train_alpha', 'val_alpha', 'train_transfer', 'val_transfer'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1, len(model_history.history['loss'])+1),
                    model_history.history['loss'])
        axs[1].plot(range(1, len(model_history.history['val_loss'])+1),
                    model_history.history['val_loss'], linestyle='--')
        axs[1].set_xticks(np.arange(
            1, len(model_history.history['loss'])+1), len(model_history.history['loss'])/10)
        axs[1].legend(['train_alpha', 'val_alpha', 'train_transfer', 'val_transfer'], loc='best')
    plt.show()


plot_model_prec_rec_history([hist1, hist2])
# transfer_model.load_weights('weights/transfer-run-1/weights-end.hdf5')

#%%

transfer_results = test_model(transfer_model)
# alpha_results = test_model(transfer_model)

#%%
print(transfer_results)
