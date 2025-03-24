import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 

import matplotlib.pyplot as plt
import pickle


def load_data():
    with open("lista_obs.pkl", "rb") as f:
        train_data = pickle.load(f)
    return train_data

# Definir RMSE como métrica personalizada
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


#Cargar los datos de entrenamiento
training_data = load_data()

# Separar en listas por columna(separamos datos de entrada y datos de salida)
input_data, output_data = map(list, zip(*training_data))

# Convertir a numpy arrays
X_train = np.copy(input_data)
Y_train = np.copy(output_data)

# Crear el modelo
modelo = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(Y_train.shape[1])  # Sin activación porque es regresión
])

# Compilar el modelo
modelo.compile(optimizer="adam",
               loss='mse',  # Minimizar error cuadrático medio
               metrics=[rmse]) #Root mean square error



# Entrenar la red
hist = modelo.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1)

#Graficar la evolución de la pérdida y RMSE
plt.figure(figsize=(12, 5))

# Gráfico de la pérdida (MSE)
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Pérdida (MSE)')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.title('Evolución de la Pérdida')
plt.legend()
plt.grid()

# Gráfico de RMSE
plt.subplot(1, 2, 2)
plt.plot(hist.history['rmse'], label='RMSE', color='orange')
plt.xlabel('Épocas')
plt.ylabel('RMSE')
plt.title('Evolución de RMSE')
plt.legend()
plt.grid()

plt.show()

# Guardar el modelo entrenado
modelo.save("WM_go2.keras")