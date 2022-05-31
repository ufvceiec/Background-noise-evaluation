from .libraries import *
from tqdm import tqdm
import gc
from .variables import *

def load_data(json_path):
    #abrir json
    with open(json_path,"r") as fp:
        data=json.load(fp)
        
        inputs_spec= np.array(data["sft_mixed"])
        targets_spec= np.array(data["sft_noise"])
        classes= np.array(data["class"])

    return inputs_spec, targets_spec, classes
    

def convert_data(path, out_wav,mix_wav_spec,scaler,i):
    #desnormalizamos para que se escuche con toda la amplitud del audio original.

    out_wav=scaler.inverse_transform(out_wav)

    mix_wav, mix_wav_phase = librosa.magphase(librosa.stft(mix_wav_spec, n_fft=N_FFT, hop_length=HOP_LENGTH))

    out_wav=librosa.feature.inverse.mel_to_stft(out_wav,sr=SAMPLE_RATE,n_fft=N_FFT)

    out_wav,magphase=librosa.magphase(out_wav)

    arrayy_sound= librosa.core.istft(
        out_wav * mix_wav_phase
        , win_length=2048, hop_length=HOP_LENGTH)

    sf.write(f'{path}_{i}.wav',arrayy_sound, SAMPLE_RATE)
    
    return(f'{path}_{i}.wav')




# def convert_MFFc(signal_mixed):
#     S_full, phase = librosa.magphase(librosa.stft(signal_mixed))

#     stft_mixed= librosa.core.stft(signal_mixed,hop_length=HOP_LENGTH,n_fft=N_FFT)

#     audio=librosa.core.istft(S_full)

#     print(np.shape(audio))

#     wavfile.write('D:\\Caracuel\\WaveNet-seaBackGroundNoise\\Code\\Building_model_MFCC\\pepe.wav', 16000, audio)



