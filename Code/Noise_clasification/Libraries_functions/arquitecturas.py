## AUTOENCODER
from matplotlib.pyplot import axis
from .libraries import *

def unet(input):

    axis_conc=3
    
    inp_lay = Input((input[1],input[2],1))

    conv1 = Conv2D(16, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(inp_lay)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = keras.layers.Dropout(0.3)(conv1)

    conv2 = Conv2D(32, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = keras.layers.Dropout(0.3)(conv2)

    conv3 = Conv2D(64, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = keras.layers.Dropout(0.3)(conv3)

    conv4 = Conv2D(128, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = keras.layers.Dropout(0.3)(conv4)

    conv5 = Conv2D(256, 5,strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = keras.layers.Dropout(0.3)(conv5)

    deconv11= keras.layers.GlobalAveragePooling2D()(conv5)
    deconv11 = Dense(4, activation="softmax")(deconv11)
    
    model = Model(inputs=inp_lay, outputs=deconv11)

    '''
    # inp_lay = Input((input.shape[1],input.shape[2],1))

    # # 1st dense layer
    # conv1=keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(inp_lay)
    # conv1=keras.layers.Dropout(0.3)(conv1)

    # # 2nd dense layer
    # conv2=keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(conv1)
    # conv2=keras.layers.Dropout(0.3)(conv2)

    # # 3rd dense layer
    # conv3=keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(conv2)
    # conv3=keras.layers.Dropout(0.3)(conv3)

    # # output layer
    # conv4=keras.layers.Dense(4, activation='softmax')(conv3)

    # model = Model(inputs=inp_lay, outputs=conv4)
    '''
    return model

def CNN_lstm(input):
    inp_lay = Input((input[1],input[2],1))
    # inp_lay = Input((input[0],input[1],1))

    conv1 = Conv2D(16, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(inp_lay)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    # conv1 = keras.layers.Dropout(0.4)(conv1)

    conv2 = Conv2D(32, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    # conv2 = keras.layers.Dropout(0.4)(conv2)

    conv3 = Conv2D(64, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    # conv3 = keras.layers.Dropout(0.4)(conv3)
    # reshape_layer=conv2[:,:,:,0]
    
    reshape_layer=tf.keras.layers.Reshape((conv2.shape[1],-1))(conv3)

    lstm1=LSTM(128,return_sequences= True)(reshape_layer)
    drop1=Dropout(0.4)(lstm1)
    lstm2=LSTM(64)(drop1)
    drop2=Dropout(0.4)(lstm2)

    #Dense layer
    dense1=Dense(128, activation="relu")(drop2)
    drop3=Dropout(0.4)(dense1)

    dense2=Dense(64, activation="relu")(drop3)
    drop4=Dropout(0.4)(dense2)
    
    dense3=Dense(32, activation="relu")(drop4)
    drop5=Dropout(0.4)(dense3)
   
    out=Dense(4, activation="softmax")(drop5)
    
    model = Model(inputs=inp_lay, outputs=out)

    return model

def NN_lstm(input):


    #2 capas LSTM 
    inp_lay = Input((input[1],input[2]))
    

    lstm1=LSTM(128)(inp_lay)
    drop1=Dropout(0.2)(lstm1)
    # lstm2=LSTM(64)(lstm1)

    #Dense layer
    dense1=Dense(128, activation="relu")(drop1)
    dense2=Dense(64, activation="relu")(dense1)
    drop2=Dropout(0.4)(dense2)

    dense3=Dense(48, activation="relu")(drop2)
    drop3=Dropout(0.4)(dense3)
   
    out=Dense(4, activation="softmax")(drop3)
    
    model = Model(inputs=inp_lay, outputs=out)


    return model
