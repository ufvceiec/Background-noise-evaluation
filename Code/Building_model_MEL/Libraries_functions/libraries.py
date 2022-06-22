import numpy as np
import matplotlib as plt
import os
import keras
from pesq import pesq
from tensorflow.keras import callbacks
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import platform
import pandas as pd
import numpy as np
# import tqdm as tqdm 
from sklearn.preprocessing import MinMaxScaler
from scipy.io import wavfile
# Only used in the custom layer for the U-Net
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate, add, BatchNormalization, Conv2DTranspose,LSTM,RepeatVector,TimeDistributed, Reshape, ReLU,Conv2D, UpSampling1D, MaxPooling1D, LeakyReLU, Lambda, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
import tensorflow 
# For the train and test split
from sklearn.model_selection import train_test_split
# For librosa
import librosa, librosa.display
import soundfile as sf
import pickle 
import matplotlib.pyplot as plt
import numpy as np 
import math
import json
import random
##########
#Image comparation
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2