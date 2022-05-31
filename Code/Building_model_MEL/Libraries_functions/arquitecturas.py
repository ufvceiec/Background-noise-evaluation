## AUTOENCODER
from matplotlib.pyplot import axis
from .libraries import *

def CNN_lstm(input):
    inp_lay = Input((input.shape[1],input.shape[2],1))
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
    
    reshape_layer=tf.keras.layers.Reshape((conv3.shape[1],-1))(conv3)
    print("reshape layer",reshape_layer.shape)
    lstm1=LSTM(128,activation='relu')(reshape_layer)
    # drop1=Dropout(0.4)(lstm1)
    repeat=RepeatVector(reshape_layer.shape[1])(lstm1)
    print("repeat vector",repeat.shape)
    lstm2=LSTM(64,activation='relu',return_sequences= True)(repeat)
    # drop2=Dropout(0.4)(lstm2)
    time_distributed=TimeDistributed(Dense(reshape_layer.shape[2]))(lstm2)

    print("time distributed", time_distributed.shape)
    undo_reshape=tf.keras.layers.Reshape((conv3.shape[1],conv3.shape[2],conv3.shape[3]))(time_distributed)
    
    deconv7 = Conv2DTranspose(64, 5, strides=2,padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(undo_reshape)
    # merge7 = concatenate([deconv7, conv4],axis=axis_conc)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = LeakyReLU(alpha=0.2)(deconv7)
    deconv7 = Dropout(0.5)(deconv7)
    

    deconv7 = Conv2DTranspose(32, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(deconv7)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = LeakyReLU(alpha=0.2)(deconv7)
    deconv7 = Dropout(0.5)(deconv7)
    

    deconv7 = Conv2DTranspose(16, 5, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(deconv7)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = LeakyReLU(alpha=0.2)(deconv7)
    deconv7 = Dropout(0.5)(deconv7)
   

    deconv7 = Conv2DTranspose(1, 2, strides=2, padding='same',kernel_regularizer=keras.regularizers.l2(0.001))(deconv7)
    deconv7 = Activation('relu')(deconv7)

    model = Model(inputs=inp_lay, outputs=deconv7)

    return model

def unet(input):

    axis_conc=3
    
    inp_lay = Input((input[0],input[1],1))

    conv1 = Conv2D(16, 5, strides=2, padding='same')(inp_lay)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(32, 5, strides=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(64, 5, strides=2, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)

    conv4 = Conv2D(128, 5, strides=2, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)


######################################

    conv5 = Conv2D(256, 5,strides=2, padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)

######################################


    deconv7 = Conv2DTranspose(128, 5, strides=2,padding='same')(conv5)
    merge7 = concatenate([deconv7, conv4],axis=axis_conc)
    deconv7 = BatchNormalization()(merge7)
    deconv7 = Dropout(0.5)(deconv7)
    deconv7 = Activation('relu')(deconv7)
    
    

    deconv8 = Conv2DTranspose(64, 5, strides=2, padding='same')(deconv7)
    merge8 = concatenate([deconv8, conv3],axis=axis_conc)
    deconv8 = BatchNormalization()(merge8)
    deconv8 = Dropout(0.5)(deconv8)
    deconv8 = Activation('relu')(deconv8)
    

    deconv9 = Conv2DTranspose(32, 5, strides=2, padding='same')(deconv8)
    merge9 = concatenate([deconv9, conv2],axis=axis_conc)
    deconv9 = BatchNormalization()(merge9)
    deconv9 = Activation('relu')(deconv9)
    

    deconv10 = Conv2DTranspose(16, 5, strides=2, padding='same')(deconv9)
    merge10 = concatenate([deconv10, conv1],axis=axis_conc)
    deconv10 = BatchNormalization()(merge10)
    deconv10 = Activation('relu')(deconv10)
   

    deconv11 = Conv2DTranspose(1, 2, strides=2, padding='same')(deconv10)
    deconv11 = Activation('relu')(deconv11)

    model = Model(inputs=inp_lay, outputs=deconv11)


#################################################################
############PAPER


    # conv1 = Conv2D(16, 5, strides=2, padding='same')(inp_lay)
    # conv1 = BatchNormalization()(conv1)
    # conv1 = LeakyReLU(alpha=0.2)(conv1)

    # conv2 = Conv2D(32, 5, strides=2, padding='same')(conv1)
    # conv2 = BatchNormalization()(conv2)
    # conv2 = LeakyReLU(alpha=0.2)(conv2)

    # conv3 = Conv2D(64, 5, strides=2, padding='same')(conv2)
    # conv3 = BatchNormalization()(conv3)
    # conv3 = LeakyReLU(alpha=0.2)(conv3)

    # conv4 = Conv2D(128, 5, strides=2, padding='same')(conv3)
    # conv4 = BatchNormalization()(conv4)
    # conv4 = LeakyReLU(alpha=0.2)(conv4)

    # conv5 = Conv2D(256, 5, strides=2, padding='same')(conv4)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = LeakyReLU(alpha=0.2)(conv5)

    # conv6 = Conv2D(512, 5, strides=2, padding='same')(conv5)
    # conv6 = BatchNormalization()(conv6)
    # conv6 = LeakyReLU(alpha=0.2)(conv6)

    # deconv7 = Conv2DTranspose(256, 5, strides=2, padding='same')(conv6)
    # deconv7 = BatchNormalization()(deconv7)
    # deconv7 = Dropout(0.5)(deconv7)
    # deconv7 = Activation('relu')(deconv7)

    # deconv8 = concatenate([deconv7, conv5],axis=axis_conc)
    # deconv8 = Conv2DTranspose(128, 5, strides=2, padding='same')(deconv8)
    # deconv8 = BatchNormalization()(deconv8)
    # deconv8 = Dropout(0.5)(deconv8)
    # deconv8 = Activation('relu')(deconv8)

    # deconv9 = concatenate([deconv8, conv4],axis=axis_conc)
    # deconv9 = Conv2DTranspose(64, 5, strides=2, padding='same')(deconv9)
    # deconv9 = BatchNormalization()(deconv9)
    # deconv9 = Dropout(0.5)(deconv9)
    # deconv9 = Activation('relu')(deconv9)

    # deconv10 = concatenate([deconv9, conv3],axis=axis_conc)
    # deconv10 = Conv2DTranspose(32, 5, strides=2, padding='same')(deconv10)
    # deconv10 = BatchNormalization()(deconv10)
    # deconv10 = Activation('relu')(deconv10)

    # deconv11 = concatenate([deconv10, conv2],axis=axis_conc)
    # deconv11 = Conv2DTranspose(16, 5, strides=2, padding='same')(deconv11)
    # deconv11 = BatchNormalization()(deconv11)
    # deconv11 = Activation('relu')(deconv11)

    # deconv12 = concatenate([deconv11, conv1],axis=axis_conc)
    # deconv12 = Conv2DTranspose(1, 5, strides=2, padding='same')(deconv12)
    # deconv12 = Activation('relu')(deconv12)

    # model = Model(inputs=inp_lay, outputs=deconv12)

#########################################
###############3SANTI

 #Downsampling
    # conv1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(inp_lay)
    # conv1 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # pool1 = MaxPooling1D(pool_size=2)(conv1)
    # conv2 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # conv2 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # pool2 = MaxPooling1D(pool_size=2)(conv2)
    # conv3 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # conv3 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # pool3 = MaxPooling1D(pool_size=4)(conv3)
    # conv4 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # conv4 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling1D(pool_size=3)(drop4)
    
    # # Bottleneck
    # conv5 = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    # conv5 = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)
    
    # # Upsampling
    # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(drop5))
    # merge6 = concatenate([drop4, up6], axis=axis_conc)
    # conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    # conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    
    # up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv6))
    # merge7 = concatenate([conv3, up7], axis=axis_conc)
    # conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    # conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    
    # up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv7))
    # merge8 = concatenate([conv2, up8], axis=axis_conc)
    # conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    # conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    
    # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv8))
    # merge9 = concatenate([conv1, up9], axis=axis_conc)
    # conv9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    
    # conv10 = Conv2D(1, 1, activation='tanh')(conv9)
    
    # model = Model(inputs=inp_lay, outputs=conv10)
    
    # return model
    return model