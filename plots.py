import matplotlib.pyplot as plt
import numpy as np

def plot_model_acc_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc'])+1),
                model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc'])+1),
                model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(
        1, len(model_history.history['acc'])+1), len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss'])+1),
                model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1),
                model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(
        1, len(model_history.history['loss'])+1), len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def plot_model_prec_rec_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc'])+1),
                model_history.history['precision'])
    axs[0].plot(range(1, len(model_history.history['val_precision'])+1),
                model_history.history['val_precision'])
    axs[0].set_title('Model Precision')
    axs[0].set_ylabel('Precision')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(
        1, len(model_history.history['precision'])+1), len(model_history.history['precision'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['recall'])+1),
                model_history.history['recall'])
    axs[1].plot(range(1, len(model_history.history['val_recall'])+1),
                model_history.history['val_recall'])
    axs[1].set_title('Model Recall')
    axs[1].set_ylabel('Recall')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(
        1, len(model_history.history['recall'])+1), len(model_history.history['recall'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
