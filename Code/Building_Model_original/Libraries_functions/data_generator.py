import tensorflow as tf
import numpy as np
from .funciones import chunk4generator
class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df,inp,
                 batch_size,
                 input_size=(1920000, 1),
                 shuffle=True):
        
        self.df = df.copy()
        self.inp = inp
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)

    def on_epoch_end(self):
        pass
    
    def __get_input(self, path):

        x = chunk4generator(path,self.inp)

        return x
    
    def __get_output(self, path):

        y = chunk4generator(path,self.inp)

        return y
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches["mixed_rute"]
        
        y_path_batch = batches['noise_rute']


        X_batch = np.asarray([self.__get_input(x) for x in path_batch])
        

        y_batch = np.asarray([self.__get_output(y) for y in y_path_batch])


        return (X_batch, y_batch)
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size