# %% [markdown]
# ## Librerias

# %%
from pydub import AudioSegment
import wave
import os
from pydub.utils import make_chunks
import glob
from .libraries import *
import ipywidgets
from .variables import SAMPLE_RATE
# %% [markdown]
# ## Variables globales
#

path_Datasets = "./data_tests/noisy_sound/"
path_TED = "./data_tests/voice_sounds/"

#know where is the .py file
def get_path():
    path = os.path.dirname(os.path.abspath(__file__))
    return path
def generar_samplerate(Ted):
    #print(get_path())
    if Ted:
        print("Estandarized TED audios")
        class_files=os.listdir(path_TED)
        for entry in tqdm(class_files):
                dataset_path = (path_TED + '/' + entry)
                with wave.open(dataset_path, "rb") as wave_file:
                        sample_rate = wave_file.getframerate()
                        if (sample_rate != SAMPLE_RATE):
                            sound = AudioSegment.from_file(
                                dataset_path, format='wav', frame_rate=sample_rate)
                            sound = sound.set_frame_rate(SAMPLE_RATE)
                            sound = sound.set_channels(1)
                            sound.export(dataset_path, format='wav')
                        else:
                            sound = AudioSegment.from_file(
                                dataset_path, format='wav', frame_rate=sample_rate)
                            sound = sound.set_channels(1)
                            sound.export(dataset_path, format='wav')
    else:
        print("Estandarized noisy audios")
        class_files = os.listdir(path_Datasets)
        for entry in (class_files):
            wav_files = os.listdir(path_Datasets + '/' + entry)
            # print(wav_files)
            for dataset in tqdm(wav_files):
                dataset_path = (
                    path_Datasets + '/' + entry + '/' + dataset)

                with wave.open(dataset_path, "rb") as wave_file:
                    sample_rate = wave_file.getframerate()
                    if (sample_rate != SAMPLE_RATE):
                        sound = AudioSegment.from_file(
                            dataset_path, format='wav', frame_rate=sample_rate)
                        sound = sound.set_frame_rate(SAMPLE_RATE)
                        sound = sound.set_channels(1)
                        sound.export(dataset_path, format='wav')
                    else:
                        sound = AudioSegment.from_file(
                            dataset_path, format='wav', frame_rate=sample_rate)
                        sound = sound.set_channels(1)
                        sound.export(dataset_path, format='wav')

# %%
def process_sudio(file_name,path_file,save_path):

    myaudio = AudioSegment.from_file(path_file+ file_name, "wav") 
    chunk_length_ms = 3000 # pydub calculates in millisec 
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of 3 sec 
    for i, chunk in enumerate(chunks): 
        # duration of chunk in sec
        if chunk.duration_seconds != 3.0:
            restante= 3.0 - chunk.duration_seconds
            # print("duracion en segundos del chunk que no llega a 3:",chunk.duration_seconds)
            # print("tiempo restante a sumar:",restante)
            audio_silent = AudioSegment.empty()
            audio_silent = AudioSegment.silent(duration=restante*2000)
            # print("aduio formado que sumar:",audio_silent.duration_seconds)
            # fill audio with 0
            chunk = chunk.append(audio_silent)
            #chunk to 3000 ms
            chunk= chunk[:3000]
        chunk_name = save_path + file_name + "_{0}.wav".format(i) 
        # print("duracion udio formateado:",chunk.duration_seconds)
        chunk.export(chunk_name, format="wav") 

def chunk_large_audio(path_file,save_path):
    py_files_m = glob.glob(f'{save_path}*.wav')

    for py_file in py_files_m:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    
    all_file_names = os.listdir(path_file)
    for each_file in (all_file_names):
        if ('.wav' in each_file):
            process_sudio(each_file,path_file,save_path)

def w_files(path1):    
    recortado=None
    #Load all the wavs (names) into a list

    WAV_list1 = os.listdir(path1)

    recortado=WAV_list1
    print('First, files are going to be load and this is the lenght of data in folder:', len(recortado))

    return recortado

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def remove_files(path):
    py_files_m = glob.glob(f'{path}*.wav')

    for py_file in py_files_m:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")

def concatenate_audios(path):
    array_voices=[]
        # for para recorrer los ficheros de noise
    class_noise = os.listdir(path)
    for entry in class_noise:
        type_noise = path+'/'+entry
        array_voices.append(type_noise)
    sound= AudioSegment.from_wav(array_voices[0])
    for i in (tnrange(1,len(array_voices))):
        sound1 = AudioSegment.from_wav(array_voices[i])
        sound = sound.append(sound1)
    sound.export(path+"generated_concatenate.wav", format="wav")

def mixingData(path_Noise,path_TED_Talks, path_save_mixed,path_save_voice, path_save_noise):
    #vaciamos la carpeta de test
    remove_files(path_save_mixed)
    remove_files(path_save_voice)
    remove_files(path_save_noise)
    
    class_noise = os.listdir(path_Noise)
    noise_array = []
    ted_array = []
    #mixed_array=[]
    # for para recorrer los ficheros de noise
    for entry in class_noise:
        type_noise = path_Noise+'/'+entry
        noise_array.append(type_noise)

    # for para recorrer los ficheros de ted
    class_ted = os.listdir(path_TED_Talks)
    for entry in class_ted:
        wav_ted = path_TED_Talks+'/'+entry
        ted_array.append(wav_ted)
    # random_number=random.sample(range(0,len(ted_array)-1),1)
    for i in (tnrange(len(ted_array))):
        sound1 = AudioSegment.from_wav(ted_array[i])
        sound1 = match_target_amplitude(sound1, -18.0)
        sound2 = AudioSegment.from_wav(noise_array[0])
        sound2 = match_target_amplitude(sound2, -22.0)
        sound2= sound2[:3000]

        combined_sounds = sound1.overlay(sound2)
        name= ted_array[i].split('/')[-1].split('.')[0]
        combined_sounds.export(
            path_save_mixed+'{}'.format(name)+'_'+'{}'.format(i)+".wav", format="wav")
        sound2.export(
            path_save_noise+'{}'.format(name)+'_'+'{}'.format(i)+".wav", format="wav")
        sound1.export(
        path_save_voice+'{}'.format(name)+'_'+'{}'.format(i)+".wav", format="wav")