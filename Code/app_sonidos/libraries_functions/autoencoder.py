from enum import auto
from statistics import mode
import tensorflow as tf
from .libraries import *

tf.compat.v1.disable_eager_execution()

def build_VAE_skipp(input, latent_space_dim, epsilonstd, alpha):
    # Compute VAE loss
    def VAE_loss(x_origin,x_out):
        error = x_origin - x_out
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        #vae_loss =(0.75*reconstruction_loss + 0.25*kl_loss)
        vae_loss =(alpha*reconstruction_loss + kl_loss)
        return vae_loss
    #build encoder
    encoder_inputs = keras.Input(shape=(np.shape(input)[1], np.shape(input)[2], 1))
    x1 = layers.Conv2D(64, 3, activation="relu",strides=2, padding="same")(encoder_inputs)
    x3= layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x1)
    x2 = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x3)
    x5= layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x2)
    x = layers.Conv2D(1024, 3, activation="relu", strides=2, padding="same")(x5)
    shape_before_bottleneck = K.int_shape(x)[1:]
    # Bottleneck
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    z_mean = layers.Dense(latent_space_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_space_dim, name="z_log_var")(x)
    def sampling(args):
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_space_dim), mean=0.,
                                stddev=epsilonstd)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    z = layers.Lambda(sampling, output_shape=(latent_space_dim,), name="z")((z_mean, z_log_var))

    # original
    # #build decoder
    num_neurons = np.prod(shape_before_bottleneck)
    x = layers.Dense(num_neurons, activation="relu")(z)
    x = layers.Reshape(shape_before_bottleneck)(x)
    x = layers.Conv2DTranspose(512, 3, activation="relu",strides=2,padding="same")(x)
    x = layers.Add()([x5,x])
    x = layers.Conv2DTranspose(256, 3, activation="relu",strides=2, padding="same")(x)
    x = layers.Add()([x2,x])
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Add()([x3,x])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Add()([x1,x])

    decoder_outputs = layers.Conv2DTranspose(1, 3,strides=2,activation="sigmoid", padding="same")(x)
    model = keras.Model(encoder_inputs, decoder_outputs, name="decoder")
    model.summary()

    model.compile(optimizer='adam', loss=VAE_loss)
    return model



class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2
        self.reconstruction_loss_weight = 1000000

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss)

    def train(self, x_train,y_train,X_val,y_val, batch_size, num_epochs):

        path="./models"
        checkpoint = [callbacks.EarlyStopping(monitor='loss',patience=1000,
        restore_best_weights=True)

#,callbacks.ModelCheckpoint(
 #           path,
  #          monitor="loss",
   #         verbose=1,
    #        save_best_only=True,
     #       mode="min"
      #  )
      ]

        history=self.model.fit(
            x_train,
            y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(X_val,y_val),
            verbose=1,
            shuffle=False,
            callbacks=[checkpoint]
        )
        return history

    def save(self,learning_rates,batch_size,num_epochs,alpha,ruta, save_folder="."):
        nombre=f"modelo_noise_lr_{learning_rates}_btch_{batch_size}_epch_{num_epochs}_alpha_{alpha}_KLloss_opt_Adam"
        
        self._create_folder_if_it_doesnt_exist(save_folder)
        # save_path = os.path.join(save_folder, f"{nombre}.h5")
        # self.model.save(save_path)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)


    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images, model=None):
        if model is not None:
            latent_representations=model.predict(images)
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def load_model_complete(X_train, alpha_arg, latent_arg, learning):
        ruta="./models/"
        batch_hyper = 42
        epoch_hyper= 3
        loss_hyper="mae"
        optimizer_hyper=tf.keras.optimizers.Adam
        nombre=None
        latent_space_dim=latent_arg
        epsilonstd=1.0
        alpha=alpha_arg
        conc_x=-1
        kernel_init = 'he_normal'
        activation_layer = "relu" 
        # Compute VAE loss
        def VAE_loss(x_origin,x_out):
            error = x_origin - x_out
            reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            #vae_loss =(0.75*reconstruction_loss + 0.25*kl_loss)
            vae_loss =(reconstruction_loss + alpha*kl_loss)
            return vae_loss
    #build encoder
        encoder_inputs = keras.Input(shape=(np.shape(X_train)[1], np.shape(X_train)[2], 1))
        x4 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(encoder_inputs)
        x4 = layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x4)
        x4_pool = layers.MaxPooling2D((2, 2))(x4)
        x1 = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x4_pool)
        x1 = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x1)
        x1_pool = layers.MaxPooling2D((2, 2))(x1)
        x3= layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x1_pool)
        x3= layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x3)
        x3_pool = layers.MaxPooling2D((2, 2))(x3)
        x2 = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x3_pool)
        x2 = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x2)
        x2_pool = layers.MaxPooling2D((2, 2))(x2)
        x5= layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x2_pool)
        x5= layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x5)
        x5_pool = layers.MaxPooling2D((2, 2))(x5)
        x = layers.Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x5_pool)
        x = layers.Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        
        # x = layers.Conv2D(1024, 3, activation="LeakyReLU", padding="same")(x5)
        shape_before_bottleneck = K.int_shape(x)[1:]
        # Bottleneck
        x = layers.Flatten()(x)
        x = layers.Dense(2048, activation="relu")(x)
        z_mean = layers.Dense(latent_space_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_space_dim, name="z_log_var")(x)
        def sampling(args):
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_space_dim), mean=0.,
                                    stddev=epsilonstd)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        z = layers.Lambda(sampling, output_shape=(latent_space_dim,), name="z")((z_mean, z_log_var))

        # original
        # #build decoder
        num_neurons = np.prod(shape_before_bottleneck)
        x = layers.Dense(num_neurons, activation="relu")(z)
        x = layers.Reshape(shape_before_bottleneck)(x)

        # x = layers.Conv2D(512, 3, activation="relu",strides=2,padding="same")(x)
        # merge = concatenate([x5_pool,x], axis = 3)
        # x = layers.Conv2DTranspose(256, 3, activation="relu",strides=2, padding="same")(merge)
        # merge = concatenate([x2_pool,x], axis = 3)
        # x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(merge)
        # merge = concatenate([x3_pool,x], axis = 3)
        # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(merge)
        # merge = concatenate([x1_pool,x], axis = 3)
        # decoder_outputs = layers.Conv2DTranspose(1, 3,strides=2,activation="tanh", padding="same")(merge)


        x = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_init)(UpSampling2D(size=2)(x))
        x = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x5,x], axis = 3)
        x= layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(UpSampling2D(size=2)(merge))
        x= layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x2,x], axis = 3)
        x= layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(UpSampling2D(size=2)(merge))
        x= layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x3,x], axis = 3)
        x= layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(UpSampling2D(size=2)(merge))
        x= layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x1,x], axis = 3)
        x= layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(UpSampling2D(size=2)(merge))
        x= layers.Conv2D(32, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x4,x], axis = 3)

        decoder_outputs = layers.Conv2D(1, 3,activation="tanh", padding="same")(merge)

        model = keras.Model(encoder_inputs, decoder_outputs, name="decoder")
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=learning), loss=VAE_loss)
        return model, "build_VAE_Skipp_complejo"

    def build_VAE_Skipp_menoscomplejo(X_train, alpha_arg, latent_arg, learning):
        ruta="./models/"
        batch_hyper = 42
        epoch_hyper= 3
        loss_hyper="mae"
        optimizer_hyper=tf.keras.optimizers.Adam
        nombre=None
        latent_space_dim=latent_arg
        epsilonstd=1.0
        alpha=alpha_arg
        conc_x=-1
        kernel_init = 'he_normal'
        activation_layer = "relu" 
        kernel_init = 'he_normal'
        activation_layer = "relu" 
            # Compute VAE loss
        def VAE_loss(x_origin,x_out):
            error = x_origin - x_out
            reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            #vae_loss =(0.75*reconstruction_loss + 0.25*kl_loss)
            vae_loss =(reconstruction_loss + alpha*kl_loss)
            return vae_loss
        #build encoder
        encoder_inputs = keras.Input(shape=(np.shape(X_train)[1], np.shape(X_train)[2], 1))
        x1 = layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(encoder_inputs)
        x1_pool = layers.Conv2D(64, 3, activation="relu",strides=2, padding="same", kernel_initializer=kernel_init)(x1)
        x3= layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x1_pool)
        x3_pool= layers.Conv2D(128, 3, activation="relu", strides=2, padding="same", kernel_initializer=kernel_init)(x3)
        x2 = layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x3_pool)
        x2_pool = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same", kernel_initializer=kernel_init)(x2)
        x5= layers.Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x2_pool)
        x5_pool= layers.Conv2D(512, 3, activation="relu", strides=2, padding="same", kernel_initializer=kernel_init)(x5)
        x = layers.Conv2D(1024, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x5_pool)
        x = layers.Conv2D(1024, 3, activation="relu", strides=2, padding="same", kernel_initializer=kernel_init)(x)
        
        # x = layers.Conv2D(1024, 3, activation="LeakyReLU", padding="same")(x5)
        shape_before_bottleneck = K.int_shape(x)[1:]
        # Bottleneck
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        z_mean = layers.Dense(latent_space_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_space_dim, name="z_log_var")(x)
        def sampling(args):
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_space_dim), mean=0.,
                                    stddev=epsilonstd)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        z = layers.Lambda(sampling, output_shape=(latent_space_dim,), name="z")((z_mean, z_log_var))

        # original
        # #build decoder
        num_neurons = np.prod(shape_before_bottleneck)
        x = layers.Dense(num_neurons, activation="relu")(z)
        x = layers.Reshape(shape_before_bottleneck)(x)

        # x = layers.Conv2D(512, 3, activation="relu",strides=2,padding="same")(x)
        # merge = concatenate([x5_pool,x], axis = 3)
        # x = layers.Conv2DTranspose(256, 3, activation="relu",strides=2, padding="same")(merge)
        # merge = concatenate([x2_pool,x], axis = 3)
        # x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(merge)
        # merge = concatenate([x3_pool,x], axis = 3)
        # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(merge)
        # merge = concatenate([x1_pool,x], axis = 3)
        # decoder_outputs = layers.Conv2DTranspose(1, 3,strides=2,activation="tanh", padding="same")(merge)


        x = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=kernel_init)(UpSampling2D(size=2)(x))
        x = Conv2D(512, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x5_pool,x], axis = 3)
        x= layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(UpSampling2D(size=2)(merge))
        x= layers.Conv2D(256, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x2_pool,x], axis = 3)
        x= layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(UpSampling2D(size=2)(merge))
        x= layers.Conv2D(128, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x3_pool,x], axis = 3)
        x= layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(UpSampling2D(size=2)(merge))
        x= layers.Conv2D(64, 3, activation="relu", padding="same", kernel_initializer=kernel_init)(x)
        merge = concatenate([x1_pool,x], axis = 3)

        decoder_outputs = layers.Conv2D(1, 3,activation="tanh", padding="same")(UpSampling2D(size=2)(merge))

        model = keras.Model(encoder_inputs, decoder_outputs, name="decoder")
        model.summary()


        model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=learning), loss=VAE_loss)
        return model, "build_VAE_Skipp_complejo"
    
    def load_model(X_train, alpha_arg, latent_arg, learning):
        ruta="./models/"
        batch_hyper = 42
        epoch_hyper= 3
        loss_hyper="mae"
        optimizer_hyper=tf.keras.optimizers.Adam
        nombre=None
        latent_space_dim=latent_arg
        epsilonstd=1.0
        alpha=alpha_arg
        conc_x=-1
    

        # Compute VAE loss
        def VAE_loss(x_origin,x_out):
            error = x_origin - x_out
            reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            #vae_loss =(0.75*reconstruction_loss + 0.25*kl_loss)
            vae_loss =(alpha*reconstruction_loss + kl_loss)
            return vae_loss
    #build encoder
        encoder_inputs = keras.Input(shape=(np.shape(X_train)[1], np.shape(X_train)[2], 1))
        x1 = layers.Conv2D(64, 3, activation="relu",strides=2, padding="same")(encoder_inputs)
        x3= layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x1)
        x2 = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x3)
        x5= layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x2)
        x = layers.Conv2D(1024, 3, activation="relu", strides=2, padding="same")(x5)
        # x = layers.Conv2D(1024, 3, activation="LeakyReLU", padding="same")(x5)
        shape_before_bottleneck = K.int_shape(x)[1:]
        # Bottleneck
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        z_mean = layers.Dense(latent_space_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_space_dim, name="z_log_var")(x)
        def sampling(args):
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_space_dim), mean=0.,
                                    stddev=epsilonstd)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        z = layers.Lambda(sampling, output_shape=(latent_space_dim,), name="z")((z_mean, z_log_var))

        # original
        # #build decoder
        num_neurons = np.prod(shape_before_bottleneck)
        x = layers.Dense(num_neurons, activation="relu")(z)
        x = layers.Reshape(shape_before_bottleneck)(x)
        x = layers.Conv2DTranspose(512, 3, activation="relu",strides=2,padding="same")(x)
        merge = concatenate([x5,x], axis = 3)
        x = layers.Conv2DTranspose(256, 3, activation="relu",strides=2, padding="same")(merge)
        merge = concatenate([x2,x], axis = 3)
        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(merge)
        merge = concatenate([x3,x], axis = 3)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(merge)
        merge = concatenate([x1,x], axis = 3)
        decoder_outputs = layers.Conv2DTranspose(1, 3,strides=2,activation="sigmoid", padding="same")(merge)
        model = keras.Model(encoder_inputs, decoder_outputs, name="decoder")
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=learning), loss=VAE_loss)
        return model

    def load_model_weights(cls, save_folder=".", name="model.h5"):
        autoencoder=cls
        autoencoder.load_weights(f"{save_folder}{name}")
        print("modelo cargado")
        return autoencoder
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters_VAE_2.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "VAE_weights2.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss\
                                                         + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)
        return kl_loss

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim,
                                  name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution,
                   name="encoder_output")([self.mu, self.log_variance])
        return x


if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()



