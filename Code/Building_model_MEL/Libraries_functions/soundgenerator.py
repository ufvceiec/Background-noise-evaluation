from .libraries import *
from .load_test_data import *
from .autoencoder import *
from .variables import *

class SoundGenerator:
    """SoundGenerator is responsible for generating audios from
    spectrograms.
    """

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = \
            self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values, True)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values, noise):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            # reshape the log spectrogram
            log_spectrogram = spectrogram[:, :, 0]
            # apply denormalisation
            denorm_log_spec = self._min_max_normaliser.denormalise(
                log_spectrogram, min_max_value["min"], min_max_value["max"])
            # log spectrogram -> spectrogram
            if(noise):
                librosa.display.specshow(denorm_log_spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)## Nos permite visualizar como un mapa de calor
                plt.title("Noise")
                plt.xlabel("Time")
                plt.ylabel("Frequency")
                plt.colorbar()
                plt.show()
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # apply Griffin-Lim
            _, phase= librosa.magphase(spec)
            #signal = librosa.istft(spec*phase, hop_length=self.hop_length)
            signal = librosa.istft(spec*min_max_value["mag_phase"], hop_length=self.hop_length)

            # append signal to "signals"
            signals.append(signal)
        return signals