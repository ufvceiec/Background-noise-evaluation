from .libraries import *
from tqdm import tqdm
import gc
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
        df_data['data'][i]= np.fromstring(df.data[i].replace('[', r'').replace('\r\n', r''), sep=' ', dtype=np.float32)
        df_data['noise_original'][i]= np.fromstring(df.noise_original[i].replace('[', r'').replace('\r\n', r''), sep=' ', dtype=np.float32)

    print('\nFinished successfully!')
    
    return df_data, df_rutas


def read_path(path_Noise,path_Mixed):
    class_noise = os.listdir(path_Noise)
    class_mixed=os.listdir(path_Mixed)
    noise_array = []
    mixed_array=[]
    # for para recorrer los ficheros de noise
    print("Leyendo los nombres de los wav de sus directorios ...")
    for entry in tqdm(class_noise):
        wav_noise = os.listdir(path_Noise+'/'+entry)
        for noise_files in wav_noise:
            type_noise = path_Noise+'/'+entry+'/'+noise_files
            noise_array.append(type_noise)

    for entry in tqdm(class_mixed):
        type_noise = path_Mixed+'/'+entry
        mixed_array.append(type_noise)

    df  = pd.DataFrame()

    df['mixed_rute']=mixed_array
    df['noise_rute']=noise_array

    return df


def read_wavs(path_Noise, path_Mixed):
    class_noise = os.listdir(path_Noise)
    class_mixed=os.listdir(path_Mixed)
    noise_array = []
    mixed_array=[]
    # for para recorrer los ficheros de noise
    print("Leyendo los nombres de los wav de sus directorios ...")
    for entry in class_noise:
        wav_noise = os.listdir(path_Noise+'/'+entry)
        for noise_files in wav_noise:
            type_noise = path_Noise+'/'+entry+'/'+noise_files
            noise_array.append(type_noise)

    for entry in class_mixed:
        type_noise = path_Mixed+'/'+entry
        mixed_array.append(type_noise)
    df=load_mixed(mixed_array,noise_array)

    df.index = pd.RangeIndex(len(df.index))

    return df

def load_mixed(mixed_list,noise_list):
    data = []
    for i in tqdm(range(len(mixed_list))):
        if os.path.isfile(mixed_list[i]):
            # Read file
            Dato = wavfile.read(mixed_list[i])
            #Dato = Dato.tolist()
            data.append([Dato[1]])
        else:
            print("Something went wrong")

    data=pd.DataFrame(data)
    df  = pd.DataFrame()

    df['data'] = data

    #liberamos memoria del ordenador sobre los arrays que no queramos utilizar
    del data
    gc.collect()
    data = []
    for i in tqdm(range(len(noise_list))):
        if os.path.isfile(noise_list[i]):
            # Read file
            Dato = wavfile.read(noise_list[i])
            #Dato = Dato.tolist()
            data.append([Dato[1]])
        else:
            print("Something went wrong")

    print(data[0])
    data=pd.DataFrame(data)
    df['noise_original']=data
    df['mixed_rute']=mixed_list
    df['noise_rute']=noise_list
    return df


def chunk4generator(path,inp):
    Dato = wavfile.read(path)
    #Dato = Dato.tolist()
    x=[Dato[1]]
    x=x[:inp]
    print(np.shape(x))
    x = np.array(x)
    x = x.T
    return x


def chunk4unet(df, col1, col2,inp):
    
    X = []
    y = []

    print("Chunking data to common format")

    for file in (df[col1]):
        # X.append(file[:16320])
        X.append(file[:inp])
        
    for file in (df[col2]):
        # y.append(file[:16320])
        y.append(file[:inp])
    return X, y

def create_csv(output):
    print("Generando csv con ruido limpiado")
    df=pd.DataFrame()
    print('\nExporting to CSV. This might take a while...\n')
    # np.set_printoptions(threshold=200000)
    np.set_printoptions(threshold=164000)
    df['clear_noise']=output
    df.to_csv('noise_clear.csv')
    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
                        precision=8, suppress=False, threshold=1000, formatter=None)
    print('\nFinished successfully!')

    return df
#normalize data 
def norm_b(data):
    max_data = np.max(data)
    min_data = np.min(data)
    if abs(min_data) > max_data:
        max_data = abs(min_data)
    data = data / max_data
    return data

def export_wav(output):
     wavfile.write(('./'+'{}'.format(1))+'_'+'{}'.format(1)+'.wav', 16000, output[33])

def plot_accuracy_loss(history, n_fold):
    fig, (acc_graph, loss_graph) = plt.subplots(2)
    fig.suptitle('Model Accuracy & Loss Results')
    acc_graph.plot(history.history['accuracy'])
    acc_graph.plot(history.history['val_accuracy'])
    acc_graph.set(xlabel='epoch', ylabel='accuracy')
    acc_graph.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
    loss_graph.plot(history.history['loss'])
    loss_graph.plot(history.history['val_loss'])
    loss_graph.set(xlabel='epoch', ylabel='loss')
    loss_graph.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    fig.savefig(os.path.join(os.getcwd(), 'results', f'Results_fold_{n_fold}.png'))