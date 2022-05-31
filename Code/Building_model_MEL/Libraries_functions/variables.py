SAMPLE_RATE = 22050
N_MFCC= 15
N_FFT=512
HOP_LENGTH=259
DURATION = 3

# Audio final length (163333 = 10sec, 81666 = 5sec, 32666 = 2sec, 4899990 = 5min= 300sec)
lenght=DURATION*1000 #ms

#number of audios to cut, it has to be equal and based on how much TED talks u are going to use
number_TED=60000
number_noise=int(number_TED/4)