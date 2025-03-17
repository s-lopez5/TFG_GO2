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

#Cargar los datos de entrenamiento
training_data = load_data()

# Separar en listas por columna(separamos datos de entrada y datos de salida)
input_data, output_data = map(list, zip(*training_data))

# Definir dimensiones de entrada y salida
dim_entrada = len(input_data[0])  
dim_salida = len(output_data[0])

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
               metrics=['mae']) #Mean Absolute Error



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

# Gráfico de MAE
plt.subplot(1, 2, 2)
plt.plot(hist.history['mae'], label='MAE', color='orange')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.title('Evolución de MAE')
plt.legend()
plt.grid()

plt.show()

# Guardar el modelo entrenado
modelo.save("WM_go2.keras")