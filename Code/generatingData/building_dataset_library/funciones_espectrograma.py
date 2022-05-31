import psutil
from .libraries import *
from tqdm import tqdm
import gc
from .variables import *

num_expected=int(SAMPLE_RATE*DURATION)
save_path="../../Datasets/Data/"

def read_wavs_mel(path_Noise, path_Mixed, json_path,leer_ruido,path_ted,power_to_db):
    class_noise = os.listdir(path_Noise)
    class_mixed=os.listdir(path_Mixed)
    class_ted=os.listdir(path_ted)
    noise_array = []
    mixed_array=[]
    ted_array=[]
    # for para recorrer los ficheros de noise
    print("Leyendo los nombres de los wav de sus directorios ...")
    if(leer_ruido==True):
        for entry in class_noise:
            type_noise = path_Noise+'/'+entry
            noise_array.append(type_noise)

        '''for entry in class_noise:
            wav_noise = os.listdir(path_Noise+'/'+entry)
            for noise_files in wav_noise:
                type_noise = path_Noise+'/'+entry+'/'+noise_files
                noise_array.append(type_noise)'''
    
    else:
        for entry in class_ted:
            type_noise = path_ted+'/'+entry
            ted_array.append(type_noise)

    for entry in class_mixed:
        type_noise = path_Mixed+'/'+entry
        mixed_array.append(type_noise)

    load_mixed_mel(mixed_array,noise_array,json_path,leer_ruido,ted_array,power_to_db)

    return 


def load_mixed_mel(mixed_list,noise_list,json_path,leer_ruido,ted_array,power_to_db):
    # restamos 50 archivos que eran de tests
    numero_archivos= number_noise-50
    # varaible para saber que archivo estamos guardando
    val_aux=0
    data= {
            "sft_mixed":[], 
            "sft_noise":[],
            "class":[]
        }
    if(leer_ruido==True):## Salen ruidos
        print("leer ruido")
        for i in tqdm(range(len(mixed_list))):
            if (os.path.isfile(mixed_list[i])) and (os.path.isfile(noise_list[i])):
                # Read file
                ### no funciona la conversion
                signal_mixed, sr_mixed = librosa.load(mixed_list[i], sr=SAMPLE_RATE)
                signal_noise, sr_noise = librosa.load(noise_list[i], sr=SAMPLE_RATE)
                mix_wav, mix_wav_phase = librosa.magphase(librosa.stft(signal_mixed, n_fft=N_FFT, hop_length=HOP_LENGTH))
                noise_wav, mix_wav_phase = librosa.magphase(librosa.stft(signal_noise, n_fft=N_FFT, hop_length=HOP_LENGTH))
                mel_spec= librosa.feature.melspectrogram(S=mix_wav,sr=SAMPLE_RATE,n_fft=N_FFT,hop_length=HOP_LENGTH)
                mel_noise_spec= librosa.feature.melspectrogram(S=noise_wav,sr=SAMPLE_RATE,n_fft=N_FFT,hop_length=HOP_LENGTH)
                if power_to_db==True:
                    mel_spec= librosa.power_to_db(mel_spec, ref=np.max)
                    mel_noise_spec= librosa.power_to_db(mel_noise_spec, ref=np.max)

                data["sft_mixed"].append(mel_spec.tolist())
                # data_phase["mag_mixed"].append(mix_wav_phase)
                data["sft_noise"].append(mel_noise_spec.tolist())
                # data_phase["mag_noise"].append(noise_wav_phase)
                #añadimos a que clase pertenece cada audio
                if i<=numero_archivos-1:
                    data["class"].append(0)
                    
                elif i>=numero_archivos and i<=(numero_archivos*2)-1:
                    data["class"].append(1)
                    
                elif i>=numero_archivos*2 and i<=(numero_archivos*3)-1:
                    data["class"].append(2)
                    
                elif i>=numero_archivos*3 and i <=(numero_archivos*4)-1:
                    data["class"].append(3)
                if psutil.virtual_memory().percent > 75:
                    print(i)
                    val_aux+=1
                    print(f"Creando Json {val_aux}...")
                    with open (f"{json_path}_{val_aux}.json", "w") as fp2:
                        json.dump(data, fp2, indent=4)
                    print("Terminado Json ...")
                    del data
                    gc.collect()
                    data= {
                            "sft_mixed":[], 
                            "sft_noise":[],
                             "class":[]
                        }
                # if(i==len(mixed_list)//4 or i==len(mixed_list)//4*2 or i==len(mixed_list)//4*3 or i==len(mixed_list)-1):
                #     print(i)
                #     val_aux+=1
                #     print(f"Creando Json {val_aux}...")
                #     with open (f"{json_path}_Json:{val_aux}.json", "w") as fp2:
                #         json.dump(data, fp2, indent=4)
                #     print("Terminado Json ...")
                #     del data
                #     gc.collect()
                #     data= {
                #             "sft_mixed":[], 
                #             "sft_noise":[],
                #              "class":[]
                #         }
            else:
                print("Something went wrong")

    else:## solo para audios normales
        for i in tqdm(range(len(mixed_list))):
            if (os.path.isfile(mixed_list[i])) and (os.path.isfile(ted_array[i])):
                # Read file
                ### no funciona la conversion
                signal_mixed, sr_mixed = librosa.load(mixed_list[i], sr=SAMPLE_RATE)
                signal_noise, sr_noise = librosa.load(ted_array[i], sr=SAMPLE_RATE)
                mix_wav, mix_wav_phase = librosa.magphase(librosa.stft(signal_mixed, n_fft=N_FFT, hop_length=HOP_LENGTH))
                noise_wav, mix_wav_phase = librosa.magphase(librosa.stft(signal_noise, n_fft=N_FFT, hop_length=HOP_LENGTH))
                mel_spec= librosa.feature.melspectrogram(S=mix_wav,sr=SAMPLE_RATE,n_fft=N_FFT,hop_length=HOP_LENGTH)
                mel_noise_spec= librosa.feature.melspectrogram(S=noise_wav,sr=SAMPLE_RATE,n_fft=N_FFT,hop_length=HOP_LENGTH)
                if power_to_db==True:
                    mel_spec= librosa.power_to_db(mel_spec, ref=np.max)
                    mel_noise_spec= librosa.power_to_db(mel_noise_spec, ref=np.max)
                
                data["sft_mixed"].append(mel_spec.tolist())
                # data_phase["mag_mixed"].append(mix_wav_phase)
                data["sft_noise"].append(mel_noise_spec.tolist())
                # data_phase["mag_noise"].append(noise_wav_phase)
                #añadimos a que clase pertenece cada audio
                if i<=numero_archivos-1:
                    data["class"].append(0)
                    
                elif i>=numero_archivos and i<=(numero_archivos*2)-1:
                    data["class"].append(1)
                    
                elif i>=numero_archivos*2 and i<=(numero_archivos*3)-1:
                    data["class"].append(2)
                    
                elif i>=numero_archivos*3 and i <=(numero_archivos*4)-1:
                    data["class"].append(3)
                if psutil.virtual_memory().percent > 75:
                    print(i)
                    val_aux+=1
                    print(f"Creando Json {val_aux}...")
                    with open (f"{json_path}_{val_aux}.json", "w") as fp2:
                        json.dump(data, fp2, indent=4)
                    print("Terminado Json ...")
                    del data
                    gc.collect()
                    data= {
                            "sft_mixed":[], 
                            "sft_noise":[],
                             "class":[]
                        }
                # if(i==len(mixed_list)//4 or i==len(mixed_list)//4*2 or i==len(mixed_list)//4*3 or i==len(mixed_list)-1):
                #     print(i)
                #     val_aux+=1
                #     print(f"Creando Json {val_aux}...")
                #     with open (f"{json_path}_{val_aux}.json", "w") as fp2:
                #         json.dump(data, fp2, indent=4)
                #     print("Terminado Json ...")
                #     del data
                #     gc.collect()
                #     data= {
                #             "mapping_mixed":[],
                #             "mapping_noise":[],
                #             "sft_mixed":[], 
                #             "sft_noise":[],
                #              "class":[]
                #         }
            else:
                print("Something went wrong")


def read_wavs_Mfcc(path_Noise, path_Mixed, json_path,leer_ruido,path_ted,power_to_db):
    class_noise = os.listdir(path_Noise)
    class_mixed=os.listdir(path_Mixed)
    class_ted=os.listdir(path_ted)
    noise_array = []
    mixed_array=[]
    ted_array=[]
    # for para recorrer los ficheros de noise
    print("Leyendo los nombres de los wav de sus directorios ...")
    if(leer_ruido==True):
        for entry in class_noise:
            type_noise = path_Noise+'/'+entry
            noise_array.append(type_noise)

        '''for entry in class_noise:
            wav_noise = os.listdir(path_Noise+'/'+entry)
            for noise_files in wav_noise:
                type_noise = path_Noise+'/'+entry+'/'+noise_files
                noise_array.append(type_noise)'''
    
    else:
        for entry in class_ted:
            type_noise = path_ted+'/'+entry
            ted_array.append(type_noise)

    for entry in class_mixed:
        type_noise = path_Mixed+'/'+entry
        mixed_array.append(type_noise)

    load_mixed_Mfcc(mixed_array,noise_array,json_path,leer_ruido,ted_array,power_to_db)

    return 

def load_mixed_Mfcc(mixed_list,noise_list,json_path,leer_ruido,ted_array,power_to_db):
    # restamos 50 archivos que eran de tests
    numero_archivos= number_noise-50
    val_aux=0
    data= {
            "sft_mixed":[], 
            "sft_noise":[],
            "class":[]
        }
    if(leer_ruido==True):## Salen ruidos
        print("leer ruido")
        for i in tqdm(range(len(mixed_list))):
            if (os.path.isfile(mixed_list[i])) and (os.path.isfile(noise_list[i])):
                # Read file
                ### no funciona la conversion
                signal_mixed, sr_mixed = librosa.load(mixed_list[i], sr=SAMPLE_RATE)
                signal_noise, sr_noise = librosa.load(noise_list[i], sr=SAMPLE_RATE)
                mel_spec= librosa.feature.mfcc(y=signal_mixed, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
                mel_noise_spec= librosa.feature.mfcc(y=signal_noise,sr=SAMPLE_RATE,n_mfcc=N_MFCC,n_fft=N_FFT,hop_length=HOP_LENGTH)
                data["sft_mixed"].append(mel_spec.tolist())
                # data_phase["mag_mixed"].append(mix_wav_phase)
                data["sft_noise"].append(mel_noise_spec.tolist())
                # data_phase["mag_noise"].append(noise_wav_phase)
                #añadimos a que clase pertenece cada audio
                if i<=numero_archivos-1:
                    data["class"].append(0)
                    
                elif i>=numero_archivos and i<=(numero_archivos*2)-1:
                    data["class"].append(1)
                    
                elif i>=numero_archivos*2 and i<=(numero_archivos*3)-1:
                    data["class"].append(2)
                    
                elif i>=numero_archivos*3 and i <=(numero_archivos*4)-1:
                    data["class"].append(3)
                if psutil.virtual_memory().percent > 75:
                    print(i)
                    val_aux+=1
                    print(f"Creando Json {val_aux}...")
                    with open (f"{json_path}_{val_aux}.json", "w") as fp2:
                        json.dump(data, fp2, indent=4)
                    print("Terminado Json ...")
                    del data
                    gc.collect()
                    data= {
                            "sft_mixed":[], 
                            "sft_noise":[],
                             "class":[]
                        }
                # if(i==len(mixed_list)//4 or i==len(mixed_list)//4*2 or i==len(mixed_list)//4*3 or i==len(mixed_list)-1):
                #     print(i)
                #     val_aux+=1
                #     print(f"Creando Json {val_aux}...")
                #     with open (f"{json_path}_{val_aux}.json", "w") as fp2:
                #         json.dump(data, fp2, indent=4)
                #     print("Terminado Json ...")
                #     del data
                #     gc.collect()
                #     data= {
                #             "sft_mixed":[], 
                #             "sft_noise":[],
                #              "class":[]
                #         }
            else:
                print("Something went wrong")
    else:## solo para audios normales
        for i in tqdm(range(len(mixed_list))):
            if (os.path.isfile(mixed_list[i])) and (os.path.isfile(ted_array[i])):
                # Read file
                ### no funciona la conversion
                signal_mixed, sr_mixed = librosa.load(mixed_list[i], sr=SAMPLE_RATE)
                signal_noise, sr_noise = librosa.load(ted_array[i], sr=SAMPLE_RATE)
                mel_spec= librosa.feature.mfcc(y=signal_mixed, sr=sr_mixed, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
                mel_noise_spec= librosa.feature.mfcc(y=signal_noise, sr=sr_noise, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH) 
                data["sft_mixed"].append(mel_spec.tolist())
                # data_phase["mag_mixed"].append(mix_wav_phase)
                data["sft_noise"].append(mel_noise_spec.tolist())
                # data_phase["mag_noise"].append(noise_wav_phase)
                #añadimos a que clase pertenece cada audio
                if i<=numero_archivos-1:
                    data["class"].append(0)
                    
                elif i>=numero_archivos and i<=(numero_archivos*2)-1:
                    data["class"].append(1)
                    
                elif i>=numero_archivos*2 and i<=(numero_archivos*3)-1:
                    data["class"].append(2)
                    
                elif i>=numero_archivos*3 and i <=(numero_archivos*4)-1:
                    data["class"].append(3)
                if psutil.virtual_memory().percent > 75:
                    print(i)
                    val_aux+=1
                    print(f"Creando Json {val_aux}...")
                    with open (f"{json_path}_{val_aux}.json", "w") as fp2:
                        json.dump(data, fp2, indent=4)
                    print("Terminado Json ...")
                    del data
                    gc.collect()
                    data= {
                            "sft_mixed":[], 
                            "sft_noise":[],
                             "class":[]
                        }
                # if(i==len(mixed_list)//4 or i==len(mixed_list)//4*2 or i==len(mixed_list)//4*3 or i==len(mixed_list)-1):
                #     print(i)
                #     val_aux+=1
                #     print(f"Creando Json {val_aux}...")
                #     with open (f"{json_path}_{val_aux}.json", "w") as fp2:
                #         json.dump(data, fp2, indent=4)
                #     print("Terminado Json ...")
                #     del data
                #     gc.collect()
                #     data= {
                #             "sft_mixed":[], 
                #             "sft_noise":[],
                #              "class":[]
                #         }
            else:
                print("Something went wrong")

def read_wavs_spectrogram(path_Noise, path_Mixed,leer_ruido,path_ted):
    class_noise = os.listdir(path_Noise)
    class_mixed=os.listdir(path_Mixed)
    class_ted=os.listdir(path_ted)
    noise_array = []
    mixed_array=[]
    ted_array=[]

    py_files_n = glob.glob(f'{save_path}Noisy_numpy/*')
    py_files_c = glob.glob(f'{save_path}clear_numpy/*')
    for py_file in py_files_n:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    
    for py_file in py_files_c:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")

    # for para recorrer los ficheros de noise
    print("Leyendo los nombres de los wav de sus directorios ...")
    if(leer_ruido==True):
        for entry in class_noise:
            type_noise = path_Noise+'/'+entry
            noise_array.append(type_noise)

        '''for entry in class_noise:
            wav_noise = os.listdir(path_Noise+'/'+entry)
            for noise_files in wav_noise:
                type_noise = path_Noise+'/'+entry+'/'+noise_files
                noise_array.append(type_noise)'''
    
    else:
        for entry in class_ted:
            type_noise = path_ted+'/'+entry
            ted_array.append(type_noise)

    for entry in class_mixed:
        type_noise = path_Mixed+'/'+entry
        mixed_array.append(type_noise)

    load_mixed_spectrogram(mixed_array,noise_array,leer_ruido,ted_array)

    return 


def is_padding_neccesary(signal):

    if(len(signal)<num_expected):
        return True
    return False

def load_mixed_spectrogram(mixed_list,noise_list,leer_ruido,ted_array):
    num_spec=0
   # restamos 50 archivos que eran de tests
    data_noisy=[]
    data_mixed=[]
    saver_noisy= Saver(f"{save_path}/Noisy_numpy/","")#path de los mezclados
    saver_clear= Saver(f"{save_path}/Clear_numpy/","")#path de los sin mezclar
    if(leer_ruido==True):## Salen ruidos
        print("leer ruido")
        for i in tqdm(range(len(mixed_list))):
            if (os.path.isfile(mixed_list[i])) and (os.path.isfile(noise_list[i])):
                # Read file
                ### no funciona la conversion
                signal_mixed, sr_mixed = librosa.load(mixed_list[i], sr=SAMPLE_RATE, mono=True)
                signal_noise, sr_noise = librosa.load(noise_list[i], sr=SAMPLE_RATE, mono=True)

                ## sft for mixed
                if is_padding_neccesary(signal_mixed):
                    num_missing_samples = num_expected - len(signal_mixed)
                    signal_mixed = Padder.right_pad(signal_mixed, num_missing_samples)

                stft_mixed= librosa.stft(signal_mixed, n_fft=N_FFT, hop_length=HOP_LENGTH)[:-1]
                spectrogram_mixed= np.abs(stft_mixed)
                log_spectrogram_mixed= librosa.amplitude_to_db(spectrogram_mixed)
                norm_min_max=MinMaxScaler()
                normalized_spectrogram_mixed=norm_min_max.fit_transform(log_spectrogram_mixed)
                data_mixed.append(normalized_spectrogram_mixed)
                ##sft for raw
                if is_padding_neccesary(signal_noise):
                    num_missing_samples = num_expected - len(signal_noise)
                    signal_noise = Padder.right_pad(signal_noise, num_missing_samples)
                stft_noise= librosa.stft(signal_noise, n_fft=N_FFT, hop_length=HOP_LENGTH)[:-1]
                spectrogram_noise= np.abs(stft_noise)
                log_spectrogram_noise= librosa.amplitude_to_db(spectrogram_noise)
                norm_min_max=MinMaxScaler()
                normalized_spectrogram_noise=norm_min_max.fit_transform(log_spectrogram_noise)
                data_noisy.append(normalized_spectrogram_noise)
                if psutil.virtual_memory().percent > 85 or i==len(mixed_list)-1:
                    print(i)
                    data_noisy=np.array(data_noisy)
                    data_mixed=np.array(data_mixed)
                    for j, (k,l) in enumerate(zip(data_mixed,data_noisy)):
                        num_spec+=1
                        saver_noisy.save_feature(k, f"{save_path}/Noisy_numpy/mixed_numpy_{num_spec}")
                        saver_clear.save_feature(l, f"{save_path}/clear_numpy/clear_numpy_{num_spec}")
                    del data_mixed
                    del data_noisy
                    gc.collect()
                    data_noisy=[]
                    data_mixed=[]

            else:
                print("Something went wrong")
    else:## solo para audios normales
        for i in tqdm(range(len(mixed_list))):
            if (os.path.isfile(mixed_list[i])) and (os.path.isfile(ted_array[i])):
                # Read file
                ### no funciona la conversion
                signal_mixed, sr_mixed = librosa.load(mixed_list[i], sr=SAMPLE_RATE)
                signal_voice, sr_noise = librosa.load(ted_array[i], sr=SAMPLE_RATE)
                if is_padding_neccesary(signal_mixed):
                    num_missing_samples = num_expected - len(signal_mixed)
                    signal_mixed = Padder.right_pad(signal_mixed, num_missing_samples)

                stft_mixed= librosa.stft(signal_mixed, n_fft=N_FFT, hop_length=HOP_LENGTH)[:-1]
                spectrogram_mixed= np.abs(stft_mixed)
                log_spectrogram_mixed= librosa.amplitude_to_db(spectrogram_mixed)
                norm_min_max=MinMaxScaler()
                normalized_spectrogram_mixed=norm_min_max.fit_transform(log_spectrogram_mixed)
                data_mixed.append(normalized_spectrogram_mixed)
                ##sft for raw
                if is_padding_neccesary(signal_voice):
                    num_missing_samples = num_expected - len(signal_voice)
                    signal_voice = Padder.right_pad(signal_voice, num_missing_samples)

                stft_voice= librosa.stft(signal_voice, n_fft=N_FFT, hop_length=HOP_LENGTH)[:-1]
                spectrogram_voice= np.abs(stft_voice)
                log_spectrogram_voice= librosa.amplitude_to_db(spectrogram_voice)
                norm_min_max=MinMaxScaler()
                normalized_spectrogram_voice=norm_min_max.fit_transform(log_spectrogram_voice)
                data_noisy.append(normalized_spectrogram_voice)
                if psutil.virtual_memory().percent > 85 or i==len(mixed_list)-1:
                    print(i)
                    data_noisy=np.array(data_noisy)
                    data_mixed=np.array(data_mixed)
                    for j, (k,l) in enumerate(zip(data_mixed,data_noisy)):
                        num_spec+=1
                        saver_noisy.save_feature(k, f"{save_path}/Noisy_numpy/mixed_numpy_{num_spec}")
                        saver_clear.save_feature(l, f"{save_path}/clear_numpy/clear_numpy_{num_spec}")
                    del data_mixed
                    del data_noisy
                    gc.collect()
                    data_noisy=[]
                    data_mixed=[]
            else:
                print("Something went wrong")

class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return 

class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return 

class Saver:
    """Saver is responsible to save an array in a file."""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path