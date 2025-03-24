import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

# Definir RMSE como métrica personalizada
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Cargar los datos de entrenamiento y validación
train_data = load_data("lista_obs.pkl")

# Separar entradas y salidas
input_data, output_data = map(list, zip(*train_data))

# Convertir a numpy arrays
X = np.array(input_data)
Y = np.array(output_data)

# Dividir en 80% entrenamiento y 20% validación
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)     #random_state se fija para que la particion de los datos sea repetible

# Cargar el conjunto de prueba externo
test_data = load_data("lista_test.pkl")
X_test, Y_test = map(np.array, zip(*test_data))

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
               metrics=[rmse])  # RMSE como métrica

# Entrenar la red con validación
hist = modelo.fit(X_train, Y_train, 
                  epochs=100, batch_size=32, 
                  validation_data=(X_val, Y_val), 
                  verbose=1)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_rmse = modelo.evaluate(X_test, Y_test, verbose=1)
print(f"Pérdida en test (MSE): {test_loss:.4f}")
print(f"RMSE en test: {test_rmse:.4f}")

# Graficar la evolución de la pérdida y RMSE
plt.figure(figsize=(12, 5))

# Gráfico de la pérdida (MSE)
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Pérdida (MSE) - Entrenamiento')
plt.plot(hist.history['val_loss'], label='Pérdida (MSE) - Validación')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.title('Evolución de la Pérdida')
plt.legend()
plt.grid()

# Gráfico de RMSE
plt.subplot(1, 2, 2)
plt.plot(hist.history['rmse'], label='RMSE - Entrenamiento', color='orange')
plt.plot(hist.history['val_rmse'], label='RMSE - Validación', color='red')
plt.xlabel('Épocas')
plt.ylabel('RMSE')
plt.title('Evolución de RMSE')
plt.legend()
plt.grid()

plt.show()

# Guardar el modelo entrenado
modelo.save("WM_go2.keras")