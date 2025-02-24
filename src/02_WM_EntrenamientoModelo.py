import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Datos de ejemplo
percepciones_T = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
acciones_T = np.array([[0.5], [0.6], [0.7]])
percepciones_T1 = np.array([[0.4, 0.5], [0.5, 0.6], [0.6, 0.7]])

# Concatenar percepciones_T y acciones_T para formar las entradas
entradas = np.concatenate((percepciones_T, acciones_T), axis=1)

# Crear el modelo
model = Sequential()
model.add(Dense(10, input_dim=entradas.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(percepciones_T1.shape[1], activation='linear'))

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
model.fit(entradas, percepciones_T1, epochs=100, batch_size=1)

# Guardar el modelo entrenado
model.save('modelo_entrenado.h5')