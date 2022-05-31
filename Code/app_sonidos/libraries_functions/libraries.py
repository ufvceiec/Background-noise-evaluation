import numpy as np
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import keras
import pandas as pd
import numpy as np
from scipy.io import wavfile
from tqdm import tnrange, tqdm_notebook
import gc
import random

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate, add, BatchNormalization, Conv2DTranspose,LSTM,RepeatVector,UpSampling2D,TimeDistributed, Reshape, ReLU,Conv2D, UpSampling1D, MaxPooling1D, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
from tensorflow.keras import metrics


from pydub import AudioSegment
# For librosa
import librosa, librosa.display
import soundfile as sf
import pickle
import matplotlib.pyplot as plt
import numpy as np 
import math
import glob
import json