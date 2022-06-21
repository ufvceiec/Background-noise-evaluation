from .libraries import *
from tqdm import tqdm
import gc
import glob
from .variables import *
from .autoencoder import *
from .soundgenerator import *

MONO = True
PATH_TO_MODEL= "../../models/"
PATH_TO_VOICES="../../Datasets/Test/voice/"
SPECTROGRAMS_SAVE_DIR = "../../Datasets/Data/tests/"
MIN_MAX_VALUES_SAVE_DIR = "../../Datasets/Data/min_max_tests/"
MIN_MAX_VALUES = "../../Datasets/Data/min_max_tests/min_max_values.pkl"
FILES_DIR = "../../Datasets/Test/mixed_sound/"
SAVE_DIR_ORIGINAL="original/"
SAVE_DIR_GENERATED="generated/"
SAVE_DIR_REAL="real/"


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

def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=4):
    num_spec_each_class = int(num_spectrograms / 4)
    print(num_spec_each_class)
    num_spec_each_class_rest = num_spec_each_class+ (num_spectrograms-num_spec_each_class*4)
    sampled_indexes_0 = random.sample(range(0,int(len(spectrograms)/4)), num_spec_each_class)
    sampled_indexes_1 = random.sample(range(int(len(spectrograms)/4),(int(len(spectrograms)/4)*2)), num_spec_each_class)
    sampled_indexes_2 = random.sample(range((int(len(spectrograms)/4)*2),(int(len(spectrograms)/4)*3)), num_spec_each_class_rest)
    sampled_indexes_3 = random.sample(range((int(len(spectrograms)/4)*3),(int(len(spectrograms)/4)*4)), num_spec_each_class)
    sampled_indexes=np.concatenate((sampled_indexes_0,sampled_indexes_1,sampled_indexes_2,sampled_indexes_3))
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    # file_paths=[i for i in sorted(file_paths, key=lambda x: (int(x.split("d")[-1].split(".")[0])))]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    return sampled_spectrogrmas, sampled_min_max_values, file_paths

def load_form_direc(dir_path):
    train=[]
    file_paths=[]
    for root, _, filenames in os.walk(dir_path):
        for file_name in sorted(filenames, key=lambda x: (int(x.split("d")[-1].split(".")[0]))):
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

def save_signals(signals, save_dir,voice_paths,type, sample_rate=22050, images=1):
    PATH="../../Datasets/model_generated_VAE/"
    py_files_m = glob.glob(f'{PATH+save_dir}*')
    for py_file in py_files_m:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    for i, signal in tqdm(enumerate(signals)):
        if images == 0:
            #lo convertimos de nuevo para guardar imagenes
            stft_mixed= librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)[:-1]
            spectrogram_mixed= np.abs(stft_mixed)
            log_spectrogram_mixed= librosa.amplitude_to_db(spectrogram_mixed)
            librosa.display.specshow(log_spectrogram_mixed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)## Nos permite visualizar como un mapa de calor
            plt.title(f"voice_{voice_paths[i]}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            save_path_image = os.path.join(PATH, "Images/"+save_dir+type+voice_paths[i] + ".png")
            plt.savefig(f'{save_path_image}')
        save_path = os.path.join(PATH, save_dir+type+voice_paths[i] + ".wav")
        sf.write(save_path, signal, sample_rate)

def read_files_order(new_path, mixed_path, real_path):
    new={
        '1':[],
        '2':[],
        '3':[],
        '4':[]
    }
    mixed={
        '1':[],
        '2':[],
        '3':[],
        '4':[]
    }
    real={
        '1':[],
        '2':[],
        '3':[],
        '4':[]
    }
    for root, _, filenames in os.walk(new_path):
        for i,file_name in enumerate(filenames):
            filepath= os.path.join(root, file_name)
            y, sr = librosa.load(filepath)
                #read file with librosa
            if (i<=len(filenames)/4-1):
                new['1'].append(y)
            elif (i>=len(filenames)/4 and i<=((len(filenames)/4)*2)-1):
                new['2'].append(y)
            elif (i>=((len(filenames)/4)*2) and i<((len(filenames)/4)*3)-1):
                new['3'].append(y)
            else :
                new['4'].append(y)
    for root, _, filenames in os.walk(mixed_path):
        for i,file_name in enumerate(filenames):
            filepath= os.path.join(root, file_name)
            y, sr = librosa.load(filepath)
                #read file with librosa
            if (i<=len(filenames)/4-1):
                mixed['1'].append(y)
            elif (i>=len(filenames)/4 and i<=((len(filenames)/4)*2)-1):
                mixed['2'].append(y)
            elif (i>=((len(filenames)/4)*2) and i<((len(filenames)/4)*3)-1):
                mixed['3'].append(y)
            else :
                mixed['4'].append(y)
    for root, _, filenames in os.walk(real_path):
        for i,file_name in enumerate(filenames):
            filepath= os.path.join(root, file_name)
            y, sr = librosa.load(filepath)
                #read file with librosa
            if (i<=len(filenames)/4-1):
                real['1'].append(y)
            elif (i>=len(filenames)/4 and i<=((len(filenames)/4)*2)-1):
                real['2'].append(y)
            elif (i>=((len(filenames)/4)*2) and i<((len(filenames)/4)*3)-1):
                real['3'].append(y)
            else :
                real['4'].append(y)
    return new, mixed,real


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

def convert_spectrograms_to_audio(spectrograms, min_max_values, noise, files_path):
    _min_max_normaliser=MinMaxNormaliser(0, 1)
    signals = []
    path_signals=[]
    contador=0
    for spectrogram, min_max_value, files_path in zip(spectrograms, min_max_values, files_path):
        # reshape the log spectrogram
        log_spectrogram = spectrogram[:, :, 0]
        # apply denormalisation
        denorm_log_spec = _min_max_normaliser.denormalise(
            log_spectrogram, min_max_value["min"], min_max_value["max"])
        if(contador<4):
            librosa.display.specshow(denorm_log_spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)## Nos permite visualizar como un mapa de calor
            name="mixed"if noise else "predicted"
            plt.title(f"{name}_Name:{files_path}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar()
            plt.show()
            contador+=1
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
    contador=0
    for spectrogram in (file_path):
        signal_noise, sr_noise = librosa.load(spectrogram, sr=SAMPLE_RATE, mono=True)

        stft_mixed= librosa.stft(signal_noise, n_fft=N_FFT, hop_length=HOP_LENGTH)[:-1]
        spectrogram_mixed= np.abs(stft_mixed)
        log_spectrogram_mixed= librosa.amplitude_to_db(spectrogram_mixed)
        if(contador<4):
            librosa.display.specshow(log_spectrogram_mixed, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)## Nos permite visualizar como un mapa de calor
            plt.title(f"voice_{spectrogram}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar()
            plt.show()
            contador+=1
        # append signal to "signals"
        signals.append(signal_noise)
        path_signals.append(spectrogram)
    return signals, path_signals

def search_voices(file_path):
    def know_number_spec(file_path):
        paths=[]
        for file in file_path:
            file_cut=file.split("d")[-1].split(".")[0]
            paths.append(file_cut)
        return paths

    def find_noise(mixed_numbers):
        voice_paths=[]
        py_files_m = glob.glob(f'{PATH_TO_VOICES}*')
        for py_file in py_files_m:
            file=(py_file.split("e")[-1].split(".")[0])
            if(file in mixed_numbers):
                voice_paths.append(py_file)
        voice_paths=[i for i in sorted(voice_paths, key=lambda x: (int(x.split("e")[-1].split(".")[0])))]
        return voice_paths
    mixed_numbers=know_number_spec(file_path)
    voice_paths=find_noise(mixed_numbers)
    print(voice_paths)
    return voice_paths
