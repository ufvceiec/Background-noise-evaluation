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
from .variables import *
from .soundgenerator import *
# %% [markdown]
# ## Variables globales
#


MONO = True
PATH_TO_MODEL= "./model/"
PATH_TO_VOICES="./data_tests/mixed_order/voice_order/"
SPECTROGRAMS_SAVE_DIR = "./data_tests/espectrogramas/"
MIN_MAX_VALUES_SAVE_DIR = "./data_tests/min_max/"
MIN_MAX_VALUES = "./data_tests/min_max/min_max_values.pkl"
FILES_DIR = "./data_tests/mixed_order/mixed_sounds/"
SAVE_DIR_ORIGINAL="./data_tests/model_generated/original/"
SAVE_DIR_GENERATED="./data_tests/model_generated/generated/"
SAVE_DIR_REAL="./data_tests/model_generated/real/"


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):

    file_paths = [file_paths[index] for index in range(len(spectrograms))]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    return spectrograms, sampled_min_max_values, file_paths

def load_form_direc(dir_path):
    train=[]
    file_paths=[]
    for root, _, filenames in os.walk(dir_path):
        for file_name in sorted(filenames, key=lambda x: (int(x.split("_")[1].split(".")[0]))):
            print(file_name)
            filepath= os.path.join(root, file_name)
            spectrogram=np.load(filepath)
            train.append(spectrogram)
            file_paths.append(filepath)
        train=np.array(train)
        train = train[..., np.newaxis]

    return train, file_paths

def load_min_max(min_max_path):
    with open (min_max_path, "rb") as f:
        min_max_values = pickle.load(f)
    return min_max_values

def save_signals(signals, save_dir,voice_paths,type, sample_rate=22050):
    py_files_m = glob.glob(f'{save_dir}*')
    for py_file in py_files_m:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, type+voice_paths[i] + ".wav")
        print(save_path)
        sf.write(save_path, signal, sample_rate)

def preprocess_files():
    ## Ejecutar solo si no tenemos los tests generados correctamente

    py_files_m = glob.glob(f'{SPECTROGRAMS_SAVE_DIR}*')
    for py_file in py_files_m:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")

    py_files_m = glob.glob(f'{MIN_MAX_VALUES_SAVE_DIR}*')
    for py_file in py_files_m:
        try:
            os.remove(py_file) 
        except OSError as e:
            print(f"Error:{ e.strerror}")


    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION,MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(N_FFT, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver= Saver(SPECTROGRAMS_SAVE_DIR,MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)

def convert_spectrograms_to_audio(spectrograms, min_max_values, noise, files_path, model_name= None):
    x=0
    _min_max_normaliser=MinMaxNormaliser(0, 1)
    signals = []
    path_signals=[]
    for spectrogram, min_max_value, files_path in zip(spectrograms, min_max_values, files_path):
        # reshape the log spectrogram
        log_spectrogram = spectrogram[:, :, 0]
        # apply denormalisation
        denorm_log_spec = _min_max_normaliser.denormalise(
            log_spectrogram, min_max_value["min"], min_max_value["max"])
        librosa.display.specshow(denorm_log_spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)## Nos permite visualizar como un mapa de calor
        name="mixed"if noise else "predicted"
        plt.title(f"{name}_Name:{files_path}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        if model_name!=None and x==0:
            plt.savefig("./data_tests/espectrograms_images/"+model_name+".png")
            x=1
        plt.show()
        # log spectrogram -> spectrogram
        spec = librosa.db_to_amplitude(denorm_log_spec)
        # apply Griffin-Lim
        _, phase= librosa.magphase(spec)
        #signal = librosa.istft(spec*phase, hop_length=self.hop_length)
        signal = librosa.istft(spec*min_max_value["mag_phase"], hop_length=HOP_LENGTH)

        # append signal to "signals"
        signals.append(signal)
        path_signals.append(files_path)
    return signals, path_signals

def spectrograms_of_voice(file_path):
    signals = []
    path_signals=[]
    for spectrogram in (file_path):
        signal_noise, sr_noise = librosa.load(spectrogram, sr=SAMPLE_RATE, mono=True)

        stft_mixed= librosa.stft(signal_noise, n_fft=N_FFT, hop_length=HOP_LENGTH)[:-1]
        spectrogram_mixed= np.abs(stft_mixed)
        log_spectrogram_mixed= librosa.amplitude_to_db(spectrogram_mixed)

        librosa.display.specshow(log_spectrogram_mixed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)## Nos permite visualizar como un mapa de calor
        plt.title(f"voice_{spectrogram}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.show()
        # append signal to "signals"
        signals.append(signal_noise)
        path_signals.append(spectrogram)
    return signals, path_signals

def search_voices(file_path):
    def know_number_spec(file_path):
        paths=[]
        for file in file_path:
            file_cut=file.split("_")[-1].split(".")[0]
            paths.append(file_cut)
        return paths

    def find_noise(mixed_numbers):
        voice_paths=[]
        py_files_m = glob.glob(f'{PATH_TO_VOICES}*')
        for py_file in py_files_m:
            file=(py_file.split("_")[-1].split(".")[0])
            if(file in mixed_numbers):
                voice_paths.append(py_file)
        return voice_paths
    mixed_numbers=know_number_spec(file_path)
    voice_paths=find_noise(mixed_numbers)
    print(voice_paths)
    return voice_paths






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