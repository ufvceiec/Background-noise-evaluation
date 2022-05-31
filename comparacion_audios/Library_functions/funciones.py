from .libraries import *
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

def load_df(name):
    
    print('\nOpening CSV...\n')
    df = pd.read_csv(name, index_col=[0])
    df.index = pd.RangeIndex(len(df.index))
    df_data=pd.DataFrame()
    df_rutas=pd.DataFrame()
    print('Reading data...\n')

    df_data['data']=df.data
    df_data['noise_original']=df.noise_original
    df_rutas['mixed_rute']=df.mixed_rute
    df_rutas['noise_rute']=df.noise_rute

    print("Preprocessing data")
    for i in tqdm(range(df_data.shape[0])):
        df_data['data'][i]= np.fromstring(df.data[i].replace('[', r'').replace('\r\n', r''), sep=' ', dtype=np.int16)
        df_data['noise_original'][i]= np.fromstring(df.noise_original[i].replace('[', r'').replace('\r\n', r''), sep=' ', dtype=np.int16)

    print('\nFinished successfully!')
    
    return df_data, df_rutas

def mean(dataset):
    return sum(dataset) / len(dataset)

def desviacion(dataset):
    return np.std(dataset)

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def signalPower(x):
    return mean(x**2)

#esta función recibe el sonido que quieres comparar sin mezcla y el sonido anterior mezclado
#en nuestro caso recibe el ruido, y de segundo parámetro el ruido mezclado con el audio
def SNRsystem(inputSig, outputSig):
    noise = outputSig-inputSig
    
    powS = signalPower(outputSig)
    powN = signalPower(noise)
    return 10*math.log10((powS-powN)/powN)