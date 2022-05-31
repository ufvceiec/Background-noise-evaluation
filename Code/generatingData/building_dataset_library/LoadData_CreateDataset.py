from time import time
from .libraries import *
from .variables import *
import random

# To see times
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

def read_from_pydub(cantidad, longitud, path,path_save,ted):
    py_files_m = glob.glob(f'{path_save}*.wav')

    for py_file in py_files_m:
        try:
            os.remove(py_file)
        except OSError as e:
            print(f"Error:{ e.strerror}")
    wav_list = w_files(path)

    i=0
    while i < int(cantidad):
        # si es ted, selecciona en orden los audios al ser todos distintos
        if (ted == True):
            x=i
        else:
            # random ya utiliza el valor por defecto del timepo del sistema
            x = random.randint(0,len(wav_list)-1)
        audio_completo=AudioSegment.empty()
        audio1=AudioSegment.from_wav(path+wav_list[x])
        #casteamos a int para que siempre sea un round hacia abajo
        longitud_audio=int(audio1.duration_seconds)*1000
        #si es ted, nos saltamos añadir audios porque ya están cortados a 10 segundos
        if(ted==True):
            audio_completo=audio1
        else:
            if(longitud_audio!=0):
                numero_repeticiones= math.ceil(longitud/longitud_audio)
                numero_repeticiones+=1
            #redondeamos a la alza para luego cortar por donde la longitud que hayamos elegido
            #si no hay que añadir audio porque la longitud que cortar es menor que la longitud del audio, nos saltamos el for
            if(numero_repeticiones==1):
                audio_completo=audio1
            else:
                audio_completo=audio_completo+audio1
                for z in range(int(numero_repeticiones)):
                    # random ya utiliza el valor por defecto del timepo del sistema
                    random_wav=random.randint(0,len(wav_list)-1)
                    audio2=AudioSegment.from_wav(path+wav_list[random_wav])
                    audio_completo=audio_completo+audio2

        audio_completo = match_target_amplitude(audio_completo, -18.0)
        audio_final=audio_completo[:longitud]
        #audio_final=audio_completo[:]
        audio_final.export(out_f = f"{path_save}sound{i}.wav", 
                format = "wav")
        i+=1
        print(f"\rAudio n {i}/{cantidad}", end="")