# %%

# %%
from Libraries_functions import *
from tqdm import tqdm

# %%
inputs1, targets1, classes1 = load_data(
    "../../Datasets/Data/data400_3.0_Noise_MFCC_NMFCC128_NFFT255_HOPLENGTH502.json")
inputs2, targets2, classes2 = load_data(
    "../../Datasets/Data/data400_3.0_Noise_MFCC_NMFCC128_NFFT255_HOPLENGTH502.json")
inputs3, targets3, classes3 = load_data(
    "../../Datasets/Data/data400_3.0_Noise_MFCC_NMFCC128_NFFT255_HOPLENGTH502.json")

inputs = np.concatenate((inputs1, inputs2, inputs3), axis=0)
#targets = np.concatenate((targets1, targets2, targets3), axis=0)
classes = np.concatenate((classes1, classes2, classes3), axis=0)


# %%
type(classes)

# %%
print("Selecciona si normalizar o no:")
print("-Normalizar: 1")
print("-No normalizar: 0")
normalizar = 1
if normalizar == 1:
    for i in range(len(inputs)):
        scalerX = MinMaxScaler()
        inputs[i] = scalerX.fit_transform(inputs[i])

    # for i in range(len(targets)):
    #     scalery = MinMaxScaler()
    #     targets[i]=scalery.fit_transform(targets[i])


# %% [markdown]
# ## Reshape para modelo de Convolucion

# %%
# X = np.reshape(inputs, (np.shape(inputs)[0], np.shape(inputs)[1],np.shape(inputs)[2],1))
# y = classes

# np.shape(X)

# %% [markdown]
# ## Reshape para modelo LSTM

# %%
X = np.reshape(inputs, (np.shape(inputs)[0], np.shape(
    inputs)[1], np.shape(inputs)[2]))
y = classes

np.shape(X)

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# %%
print("Shape X: ", np.shape(X_train))

# %%
print("Shape y:", np.shape(y_train))

# %%
y_train

# %%
ruta = "./models/"
learning_rates = 0.0005
batch_hyper = 32
epoch_hyper = 80
loss_hyper = "sparse_categorical_crossentropy"
optimizer_hyper = tf.keras.optimizers.Adam
nombre = None

# %% [markdown]
# ## Model fit of conv2d

# %%
# model = unet(X)
# opti = optimizer_hyper(learning_rate=0.0001)
# model.compile(optimizer=opti, loss=loss_hyper, metrics=["accuracy"])
# model.summary()

# %% [markdown]
# ## Model fit of CNN + LSTM

# %%
model = CNN_lstm(X)
opti = optimizer_hyper(learning_rate=learning_rates)
model.compile(optimizer=opti, loss=loss_hyper, metrics=["accuracy"])
model.summary()

# %% [markdown]
# ## Model fit of LSTM

# %%
# model = RNN_lstm(X)
# opti = optimizer_hyper(learning_rate=0.0001)
# model.compile(optimizer=opti, loss=loss_hyper, metrics=["accuracy"])
# model.summary()

# %%
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# %%
path = "./models"
checkpoint = [callbacks.EarlyStopping(monitor='val_accuracy', patience=1000,
                                      restore_best_weights=True),
              callbacks.ModelCheckpoint(
    path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max"
)]

history = model.fit(
    X_train,
    y_train,
    epochs=epoch_hyper,
    batch_size=batch_hyper,
    validation_data=(X_val, y_val),
    verbose=1,
    shuffle=False,
    callbacks=[checkpoint]
)
if(optimizer_hyper == tf.keras.optimizers.Adam):
    nombre = f"modelo_noise_lr_{learning_rates}_btch_{batch_hyper}_epch_{epoch_hyper}_loss_{loss_hyper}_opt_Adam"
    model.save(f"{ruta}{nombre}.h5")
else:
    nombre = f"modelo_noise_lr_{learning_rates}_btch_{batch_hyper}_epch_{epoch_hyper}_loss_{loss_hyper}_opt_SGD.h5"
    model.save(f"{ruta}{nombre}.h5")

# %%
plot_accuracy_loss(history, nombre)

# %%
Y_pred = model.predict(X_val)
Y_pred = np.argmax(Y_pred, axis=1)
Confusion_matrix = confusion_matrix(y_val, Y_pred, labels=[0, 1, 2, 3])

# %%
disp = ConfusionMatrixDisplay(confusion_matrix=Confusion_matrix, display_labels=[
                              'Aircrafts', 'Lluvia', 'Trafico', 'Viento'])

# %%
disp.plot()
plt.savefig(f"./Images_confusion/confusion_matrix_{nombre}.png")
