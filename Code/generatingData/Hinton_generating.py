from tqdm import tqdm
from building_dataset_library import *

path_hinton_data="/media/hinton/HINTON_DATA/Javier/classification/"

path_noise_order = "/media/hinton/HINTON_DATA/Javier/classification/Datasets/Noise/All_noises_order/"
path_mezclado_large = "/media/hinton/HINTON_DATA/Javier/classification/Datasets/Audio_mezclado_large/"
path_voice_order = "/media/hinton/HINTON_DATA/Javier/classification/Datasets/TED/TED_order/"

mfcc = int(input("Enter 1 for mfcc, 2 for MEL: "))

if(mfcc == 2):

    print("1- Json with noise")
    print("2- Json with voice")
    valx = int(input())

    print("1- power to db")
    print("2- no power to db")
    valy = int(input())

    if(valx == 1):
        if(valy == 1):
            read_wavs_mel(path_noise_order, path_mezclado_large,
                            f"{path_hinton_data}/Datasets/Data/data{number_TED}_{lenght/1000}_Noise_powerToDB_mel_NMFCC{N_MFCC}_NFFT{N_FFT}_HOPLENGTH{HOP_LENGTH}", True, path_voice_order, True)
        else:
            read_wavs_mel(path_noise_order, path_mezclado_large,
                            f"{path_hinton_data}/Datasets/Data/data{number_TED}_{lenght/1000}_Noise_NopowerToDB_mel_NMFCC{N_MFCC}_NFFT{N_FFT}_HOPLENGTH{HOP_LENGTH}", True, path_voice_order, False)
    else:
        if(valy == 1):
            read_wavs_mel(path_noise_order, path_mezclado_large,
                            f"{path_hinton_data}/Datasets/Data/data{number_TED}_{lenght/1000}_Voice_powerToDB_mel_NMFCC{N_MFCC}_NFFT{N_FFT}_HOPLENGTH{HOP_LENGTH}", False, path_voice_order, True)
        else:
            read_wavs_mel(path_noise_order, path_mezclado_large,
                            f"{path_hinton_data}/Datasets/Data/data{number_TED}_{lenght/1000}_Voice_NopowerToDB_mel_NMFCC{N_MFCC}_NFFT{N_FFT}_HOPLENGTH{HOP_LENGTH}", False, path_voice_order, False)
else:
    print("1- Json with noise")
    print("2- Json with voice")
    valx = int(input())

    if(valx == 1):
        read_wavs_Mfcc(path_noise_order, path_mezclado_large,
                        f"{path_hinton_data}/Datasets/Data/data{number_TED}_{lenght/1000}_Noise_MFCC_NMFCC{N_MFCC}_NFFT{N_FFT}_HOPLENGTH{HOP_LENGTH}", True, path_voice_order, False)
    else:
        read_wavs_Mfcc(path_noise_order, path_mezclado_large,
                        f"{path_hinton_data}/Datasets/Data/data{number_TED}_{lenght/1000}_Voice_MFCC_NMFCC{N_MFCC}_NFFT{N_FFT}_HOPLENGTH{HOP_LENGTH}", False, path_voice_order, True)

