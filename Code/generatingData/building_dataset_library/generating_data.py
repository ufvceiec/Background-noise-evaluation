# %% [markdown]
# ## Librerias

# %%
from pydub import AudioSegment
import wave
import os
from pydub.utils import make_chunks
from tqdm import tqdm
import glob

from .variables import SAMPLE_RATE
# %% [markdown]
# ## Variables globales
#

path_Datasets = "../../Datasets/Noise"
path_TED = "../../Datasets/TED/"
path_thunder_storm = "../../Datasets/LLuvia_original/"
path_TED_chunk="../../Datasets/TED/PRUEBAS/"
name_ChunkedLarge="Chunked_large"
name_AllnoisesOrder="All_noises_order"
name_pruebas="PRUEBAS"

# %%


def generar_samplerate(Ted):
    if Ted:
        class_files=os.listdir(path_TED)
        for entry in (class_files):
            if (name_pruebas==entry):
                wav_files = os.listdir(path_TED + '/' + entry)
                for dataset in tqdm(wav_files):
                    dataset_path = (
                        path_TED + '/' + entry + '/' + dataset)
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
        class_files = os.listdir(path_Datasets)
        for entry in (class_files):
            wav_files = os.listdir(path_Datasets + '/' + entry)
            # print(wav_files)
            for dataset in tqdm(wav_files):
                dataset_path = (
                    path_Datasets + '/' + entry + '/' + dataset)
                # print(dataset_path)
                if(entry==name_ChunkedLarge or entry==name_AllnoisesOrder):
                    pass
                else:
                    # for files in dataset_path:
                    #     file_path = (path_Datasets + '/' + entry +
                    #                 '/' + dataset + '/' + files)
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
    chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
    for i, chunk in enumerate(chunks): 
        # chunk_name = '../../Datasets/Noise/LLuvia_truenos/' + file_name + "_{0}.wav".format(i)
        chunk_name = save_path + file_name + "_{0}.wav".format(i) 
        chunk.export(chunk_name, format="wav") 

def chunk_large_audio(path_file,save_path):
    py_files_m = glob.glob(f'{save_path}*.wav')

    for py_file in py_files_m:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    
    all_file_names = os.listdir(path_file)
    for each_file in tqdm(all_file_names):
        if ('.wav' in each_file):
            process_sudio(each_file,path_file,save_path)

#delete first 15 seconds and last 15 seconds of audio files in folder
def delete_first_last_15_seconds():
    all_file_names = os.listdir(path_TED_chunk)
    for each_file in tqdm(all_file_names):
        if ('.wav' in each_file):
            sound = AudioSegment.from_file(path_TED_chunk+ each_file, "wav")
            sound = sound[15000:len(sound)-15000]
            sound.export(path_TED+"TED_Chunked/"+each_file, format="wav")
