from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib as plt
import os
import keras
from tensorflow.keras import callbacks
import platform
import pandas as pd
import numpy as np
import tqdm as tqdm 
from sklearn.preprocessing import MinMaxScaler
from scipy.io import wavfile
# Only used in the custom layer for the U-Net
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate, add, BatchNormalization, Conv2DTranspose,LSTM
from tensorflow.keras.layers import Conv2D, UpSampling1D, MaxPooling1D, LeakyReLU
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
import tensorflow 
# For the train and test split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay,confusion_matrix
# For librosa
import librosa, librosa.display
import soundfile as sf
import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import math
import json
import random
