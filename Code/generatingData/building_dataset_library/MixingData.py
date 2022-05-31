from pydub import AudioSegment
import os
from tqdm import tnrange
import gc

from .libraries import *
from .LoadData_CreateDataset import *
from .variables import *

def chunk(df, col1, col2):
    
    X = []
    y = []

    print("Chunking data to common format")

    for file in (df[col1]):
        X.append(file[:16320])
        
    for file in (df[col2]):
        y.append(file[:16320])
    
    return X, y

def mixingData(path_Noise,path_TED_Talks):
    #vaciamos la carpeta de test
    py_files_m = glob.glob('D:/Caracuel/WaveNet-seaBackGroundNoise/Datasets/Test/mixed_sound/*.wav')
    py_files_n = glob.glob('D:/Caracuel/WaveNet-seaBackGroundNoise/Datasets/Test/noise/*.wav')
    py_files_v = glob.glob('D:/Caracuel/WaveNet-seaBackGroundNoise/Datasets/Test/voice/*.wav')

    py_mixed = glob.glob('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Audio_mezclado_large\\*.wav')
    py_noise = glob.glob('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\All_noises_order\\*.wav')
    py_voice = glob.glob('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\TED\\TED_order\\*.wav')

    for py_file in py_files_m:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    for py_file in py_files_n:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    for py_file in py_files_v:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")


    for py_file in py_mixed:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")

    for py_file in py_noise:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")

    for py_file in py_voice:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")

    random_num=[]
    class_noise = os.listdir(path_Noise)
    noise_array = []
    ted_array = []
    #mixed_array=[]
    # for para recorrer los ficheros de noise
    for entry in class_noise:
        wav_noise = os.listdir(path_Noise+'/'+entry)
        for noise_files in wav_noise:
            type_noise = path_Noise+'/'+entry+'/'+noise_files
            noise_array.append(type_noise)

    # for para recorrer los ficheros de ted
    class_ted = os.listdir(path_TED_Talks)
    for entry in class_ted:
        wav_ted = path_TED_Talks+'/'+entry
        ted_array.append(wav_ted)

    for i in range(4):#generamos 4 numeros aleatorios
        if i==0:
            random_num.append(random.sample(range(0,number_noise-1),50))
        elif i ==1:
            
            random_num.append(random.sample(range(number_noise,(number_noise*2)-1),50))
        elif i ==2:
            
            random_num.append(random.sample(range((number_noise*2),(number_noise*3)-1),50))
        else:
            
            random_num.append(random.sample(range(number_noise*3,(number_noise*4)-1),50))
            
    random_num=random_num[0]+random_num[1]+random_num[2]+random_num[3]
    
    for i in (tnrange(len(noise_array))):
        sound1 = AudioSegment.from_wav(ted_array[i])
        sound2 = AudioSegment.from_wav(noise_array[i])

        combined_sounds = sound1.overlay(sound2)
        #si es el audio elegido lo cortamos y lo guardamos en test
        if i in random_num:
            combined_sounds.export(
            "D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Test\\mixed_sound\\mixed_sound"+'{}'.format(i)+".wav", format="wav")
            sound2.export(
            "D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Test\\noise\\sound"+'{}'.format(i)+".wav", format="wav")
            sound1.export(
            "D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Test\\voice\\voice"+'{}'.format(i)+".wav", format="wav")
        else:
            combined_sounds.export(
                "D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Audio_mezclado_large\\mixedSound"+'{}'.format(i)+".wav", format="wav")
            sound2.export(
                "D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\All_noises_order\\sound_"+'{}'.format(i)+".wav", format="wav")
            sound1.export(
            "D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\TED\\TED_order\\voice"+'{}'.format(i)+".wav", format="wav")
   
   ##########################################################
   
'''#  mixed_array.append("D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Audio_mezclado_large\\mixedSound"+'{}'.format(i)+".wav")

    #liberamos memoria del ordenador sobre los arrays que no queramos utilizar
    # del #ted_array
    # gc.collect()
    # df=load_mixed(mixed_array,noise_array)

    # df.index = pd.RangeIndex(len(df.index))

    # df = preparing_df(df)

    # # df=df.drop(labels='Unnamed: 0',axis=1)
    
    # df=df.drop(labels='name',axis=1)

    # # for i in tnrange(len(df)):
    # #     df.data[i] = norm_b(df.data[i])
    # #     df.noise_original[i] = norm_b(df.noise_original[i])

    # # for i in tnrange(len(df)):
    # #     df.data[i] = rounding(df.data[i])
    # #     df.noise_original[i] = rounding(df.noise_original[i])

    # export_files(df, 0,1)

    #save_work(df,"dataframe")
'''

