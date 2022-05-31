
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

from libraries import *

# Try this to disable warnings
import warnings
warnings.filterwarnings('ignore')

# To see times

# %%


def save_work(df, name):
    
    print('\nExporting to CSV. This might take a while...\n')
    #np.set_printoptions(threshold=200000)
    np.set_printoptions(threshold=164000)
    df.to_csv(name+'.csv')
    np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)
    print('\nFinished successfully!')

# %% [markdown]
# ### Export group of the 4 files

# %% [markdown]
# Here are going to be exported the group of the 4 files composed by: Original, random noise, incompleted and low quality.
# The starting file is a random one and it will recive how many from them will be exported.

# %%


def export_groups(df, num):

    if num != 0:

        r_fi = random.randint(0, len(df))

        for i in tnrange(0, num):
            wavfile.write(('outputs/Noise/'+'{}'.format(i)) +
                          '_original.wav', 16000, df['data'][i+r_fi])
            wavfile.write(('outputs/Noise/'+'{}'.format(i)) +
                          '_random.wav', 16000, df['random_data'][i+r_fi])
            wavfile.write(('outputs/Noise/'+'{}'.format(i)) +
                          '_incompl.wav', 16000, df['incompleted_data'][i+r_fi])
            wavfile.write(('outputs/Noise/'+'{}'.format(i)) +
                          '_low_q.wav', 8000, df['low_quality_data'][i+r_fi])

# %% [markdown]
# ### Export to WAV files

# %% [markdown]
# We can also export some of the files back into WAV files but with the new lengths, shapes, etc. We are going to use the wavfile.write() function.
#
# Lets export some of them:

# %%


def export_files(df, name, valor):
    path1="D:/Caracuel/WaveNet-seaBackGroundNoise/Datasets/TED/voice_chunked"
    option = -1
    WAV_list1 = os.listdir(path1)
    while option != 0 or option != 1 or option != 2:

        if valor == 0:
            if(name == 1):
                for i in tnrange(len(df)):                 
                    #wavfile.write(('Exported/test'+'{}'.format(r_fi))+'.wav', df['sample_rate'][r_fi], df['data'][r_fi])
                    # wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\Chunked\\Aircrafts\\' +
                    #               '{}'.format(i))+'_'+'{}'.format(i)+'.wav', 16000, df['data'][i])
                    wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\Chunked\\Aircrafts\\' +
                                   '{}'.format(i))+'_'+'{}'.format(i)+'.wav', 16000, df['data'][i])                
            elif(name == 2):
                for i in tnrange(len(df)):
                    # r_fi = random.randint(0, len(df)-1)
                    #wavfile.write(('Exported/test'+'{}'.format(r_fi))+'.wav', df['sample_rate'][r_fi], df['data'][r_fi])
                    # wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\Chunked\\Lluvia\\' +
                    #               '{}'.format(i))+'_'+'{}'.format(r_fi)+'.wav', 16000, df['data'][r_fi])
                    wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\Chunked\\Lluvia\\' +
                                   '{}'.format(i))+'_'+'{}'.format(i)+'.wav', 16000, df['data'][i])                 
            elif(name == 3):
                for i in tnrange(len(df)):
                    # r_fi = random.randint(0, len(df)-1)
                    #wavfile.write(('Exported/test'+'{}'.format(r_fi))+'.wav', df['sample_rate'][r_fi], df['data'][r_fi])
                    # wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\Chunked\\Trafico\\' +
                    #               '{}'.format(i))+'_'+'{}'.format(r_fi)+'.wav', 16000, df['data'][r_fi])
                    wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\Chunked\\Trafico\\' +
                                   '{}'.format(i))+'_'+'{}'.format(i)+'.wav', 16000, df['data'][i])                  
            elif(name == 4):
                for i in tnrange(len(df)):
                    # r_fi = random.randint(0, len(df)-1)
                    #wavfile.write(('Exported/test'+'{}'.format(r_fi))+'.wav', df['sample_rate'][r_fi], df['data'][r_fi])
                    # wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\Chunked\\Viento\\' +
                    #               '{}'.format(i))+'_'+'{}'.format(r_fi)+'.wav', 16000, df['data'][r_fi])
                    wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\Noise\\Chunked\\Viento\\' +
                                   '{}'.format(i))+'_'+'{}'.format(i)+'.wav', 16000, df['data'][i])
            elif(name == 5):
                longitud=len(WAV_list1)
                for i in tnrange(len(df)):
                    # r_fi = random.randint(0, len(df)-1)
                    #wavfile.write(('Exported/test'+'{}'.format(r_fi))+'.wav', df['sample_rate'][r_fi], df['data'][r_fi])
                    # wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\TED\\voice_chunked\\' +
                    #               '{}'.format(i))+'_'+'{}'.format(r_fi)+'.wav', 16000, df['data'][r_fi])
                    wavfile.write(('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Datasets\\TED\\voice_chunked\\' +
                                  '{}'.format(longitud+i))+'_'+'{}'.format(i)+'.wav', 16000, df['data'][i])                   
            break

        elif valor == 1:

            for i in range(10):
                #wavfile.write(('Exported/test'+'{}'.format(i))+'.wav', df['sample_rate'][i], df['data'][i])
                wavfile.write(('D:/Caracuel/WaveNet-seaBackGroundNoise/Datasets/Outputs/10Exported/'+'{}'.format(i)
                               )+'_'+'{}'.format(i)+'.wav', 16000, df['data'][i])
            break

        elif valor == 2:

            print('\nEnter the file id you want to export:')
            fia = input()
            fi = int(fia)

            wavfile.write(('D:/Caracuel/WaveNet-seaBackGroundNoise/Datasets/Outputs/Specific/'+'{}'.format(fi)
                           )+'_'+'{}'.format(fi)+'.wav', 16000, df['data'][fi])
            break

        else:
            print('Not a valid option, try again...\n\n')

    print('Exported successfully')
