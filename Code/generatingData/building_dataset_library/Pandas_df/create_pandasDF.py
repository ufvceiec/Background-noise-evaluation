import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

from libraries import *

def load_data(WAV_list, path):    
    
    
    print('\nReading audio files...\n')

    # Save data into another list
    data = [0]
    x=0
    for i in (tnrange(499)):
    #for i in (tnrange(2000)):
        
        if(x==len(WAV_list)):
            x=0
        fname = path +WAV_list[x]

        if os.path.isfile(fname):
            # Read file
            Dato = wavfile.read(fname)
            #Dato = Dato.tolist()
            data.append([Dato])
        else:
            print("Something went wrong")
        x=x+1
    data = pd.DataFrame(data)
    #data = data.T
    data = data.iloc[1:].reset_index(drop=True)
    pepe=pd.DataFrame()
    pepe['data']=data

    # We first create the empty dataframe
    col_names =  ['name', 'sample_rate', 'original_length', 'final_length', 'data']
    df  = pd.DataFrame(columns = col_names)

    # We create empty lists to load the data and then save it into the dataframe
    all_data = []
    all_rates = []
    all_lengths = []

    for i in range (len(pepe)):
        all_data.append(pepe['data'][i][0][1])
        all_rates.append(pepe['data'][i][0][0])
        all_lengths.append(len(pepe['data'][i][0][1]))

    df['name'] = WAV_list[0]
    df['data'] = all_data
    df['sample_rate'] = all_rates
    df['original_length'] = all_lengths

    return df

def load_mixed(mixed_list,noise_list):
#######################

    data = []
    for i in tnrange(len(mixed_list)):
        if os.path.isfile(mixed_list[i]):
            # Read file
            Dato = wavfile.read(mixed_list[i])
            #Dato = Dato.tolist()
            data.append([Dato])
        else:
            print("Something went wrong")

    data = pd.DataFrame(data)
    #data = data.T
    #data = data.iloc[1:].reset_index(drop=True)
    
    pepe=pd.DataFrame()
    pepe['data']=data

    # We first create the empty dataframe
    col_names =  ['name', 'sample_rate', 'original_length', 'final_length', 'data']
    df  = pd.DataFrame(columns = col_names)

    # We create empty lists to load the data and then save it into the dataframe
    all_data = []
    all_rates = []
    all_lengths = []
    for i in range (len(pepe)):
        all_data.append(pepe['data'][i][1])
        all_rates.append(pepe['data'][i][0])
        all_lengths.append(len(pepe['data'][i][1]))
    
    data = pd.DataFrame(data)
    #data = data.T
    #data = data.iloc[1:].reset_index(drop=True)
    pepe=pd.DataFrame()
    pepe['data']=data

    df['name'] = mixed_list[0]
    df['data'] = all_data
    df['sample_rate'] = all_rates
    df['original_length'] = all_lengths

    #liberamos memoria del ordenador sobre los arrays que no queramos utilizar
    del data
    del pepe
    del all_data
    gc.collect()

    #noise read
    data = []

    for i in tnrange(len(noise_list)):
        if os.path.isfile(noise_list[i]):
            # Read file
            Dato = wavfile.read(noise_list[i])
            #Dato = Dato.tolist()
            data.append([Dato])
        else:
            print("Something went wrong")

    data = pd.DataFrame(data)
    #data = data.T
    #data = data.iloc[1:].reset_index(drop=True)
    pepe=pd.DataFrame()
    pepe['data']=data

    all_data = []

    for i in range (len(pepe)):
        all_data.append(pepe['data'][i][1])

    df['noise_original']=all_data
    df['mixed_rute']=mixed_list
    df['noise_rute']=noise_list

    return df



def load_df(name):
    
    print('\nOpening CSV...\n')
    df = pd.read_csv(name, index_col=[0])
    df.index = pd.RangeIndex(len(df.index))
    tqdm.pandas()
    df.progress_apply(lambda x: x)
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

def get_lenghts(df):
    
    min_len = df.original_length.min()
    max_len = df.original_length.max()
    avg_len = np.mean(df.original_length)

    print('Max Length: ', max_len, '\nAverage Length: ', int(avg_len), '\nMin Length: ', min_len);

def rounding(data):
    for x in range(0, data.size):
        data[x] = round(data[x], 3)
    return data


# We are going to chunk all the files in the same length (10 seconds).
# 
# 10 secs = 163333 frames.
# 
# We are going to remove the first 27 seconds (Ted Intro) and take the next 10 seconds, which has actual speech.
# 
# If we want a different final length we just have to modify the values from the box below.


# Main parameters:


def generic_chunk(df, beg, f_len,ted):
    
    new = df[['data']].copy()

    if (ted==False):
        beg = random.randint(0, beg)
    
    new['data'] = [col[beg:beg+f_len] for col in tqdm_notebook(new.data)]
    
    return new

def chunk(df):
    
    # How much we want to cut from the beginning (441000 = 27 seconds):
    beg = 441000

    # How long we want the final length (163333 = 10 seconds)
    f_len = 163333
    
    df['data'] = [col[beg:beg+f_len] for col in df.data]
    
    # In case some file was shorten than the length we set, the remaining empty part is going be fill with 0's
    df['data'] = [np.pad(col, (0, f_len-len(col)), 'constant') for col in df.data if len(df.data) < f_len]
    
    df['final_length'] = [len(col) for col in df.data]
    
    return df

# To take the second segment for a bigger dataset.
# From second 37 to 47

def chunk_2(df):
    
    # How much we want to cut from the beginning (604333 = 37 seconds):
    beg = 604333

    # How long we want the final length (163333 = 10 seconds)
    f_len = 163333
    
    df['data'] = [col[beg:beg+f_len] for col in df.data]
    
    # In case some file was shorten than the length we set, the remaining empty part is going be fill with 0's
    df['data'] = [np.pad(col, (0, f_len-len(col)), 'constant') for col in df.data if len(df.data) < f_len]
    
    df['final_length'] = [len(col) for col in df.data]
    
    return df

# To take the third segment for a bigger dataset.
# From second 47 to 57


def chunk_3(df):
    
    # How much we want to cut from the beginning (604333 = 37 seconds):
    beg = 767666
    
    # How long we want the final length (163333 = 10 seconds)
    f_len = 163333
    
    df['data'] = [col[beg:beg+f_len] for col in df.data]
    
    # In case some file was shorten than the length we set, the remaining empty part is going be fill with 0's
    df['data'] = [np.pad(col, (0, f_len-len(col)), 'constant') for col in df.data if len(df.data) < f_len]
    
    df['final_length'] = [len(col) for col in df.data]
    
    return df


# ### Chunking the DF to export

def preparing_df(df):
    if 'sample_rate' in df:
        df = df.drop(['sample_rate'], axis=1)
        
    if 'original_length' in df:
        df = df.drop(['original_length'], axis=1)
        
    if 'final_length' in df:
        df = df.drop(['final_length'], axis=1)
        
    return df

