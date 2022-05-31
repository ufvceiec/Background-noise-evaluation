from .libraries import *
from tqdm import tqdm
import gc

def plot_accuracy_loss(history, name):
    last_val_loss=history.history['val_loss'][-1]
    fig = plt.figure()
    loss_graph=fig.add_subplot(1,1,1)
    fig.suptitle('Model MSE & Loss Results')
    loss_graph.plot(history.history['loss'])
    loss_graph.plot(history.history['val_loss'])
    loss_graph.set(xlabel='epoch', ylabel='loss')
    loss_graph.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    fig.savefig(f'Results_ValLoss_{name}.png')