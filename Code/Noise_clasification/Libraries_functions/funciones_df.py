from cProfile import label
from .libraries import *
from tqdm import tqdm
import gc

def plot_accuracy_loss(history, name):
    fig, axs = plt.subplots(2)
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    fig.savefig(f'./Images_losses/Results_ValLoss_{name}.png')

    # last_val_loss=history.history['val_loss'][-1]
    # fig = plt.figure()
    # loss_graph=fig.add_subplot(1,1,1)
    # fig.suptitle('Model MSE & Loss Results')
    # loss_graph.plot(history.history['loss'])
    # loss_graph.plot(history.history['val_loss'])
    # loss_graph.set(xlabel='epoch', ylabel='loss')
    # loss_graph.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    