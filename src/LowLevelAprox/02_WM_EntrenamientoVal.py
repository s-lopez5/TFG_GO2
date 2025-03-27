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

# Cierra la pestaña del grafico con "ESC"
def on_key(event):
    if event.key == "escape":
        plt.close()

# Definir RMSE como métrica personalizada
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def build_model():
    # Crear el modelo
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(Y_train.shape[1])  # Sin activación porque es regresión
    ])

    large_model = tf.keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(Y_train.shape[1])
    ])


    # Compilar el modelo
    model.compile(optimizer="adam",
                loss="mse",  # Minimizar error cuadrático medio
                metrics=["mae", "mse"])  # mean absolute error, error cuadratico medio
    
    return model

def plot_hist():

    fig = plt.figure(figsize=(12, 5))

    # Activa la salida con escape
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Gráfico de MAE
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["mae"], label="MAE - Entrenamiento")
    plt.plot(hist.history["val_mae"], label="MAE - Validación")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error [MPG]")
    plt.title("MAE (Mean Abs Error)")
    plt.legend()
    plt.grid()

    # Gráfico de MSE
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["mse"], label="MSE - Entrenamiento", color="orange")
    plt.plot(hist.history["val_mse"], label="MSE - Validación", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error [$MPG^2$]")
    plt.title("MSE (Mean Square Error)")
    plt.legend()
    plt.grid()

    plt.show()
    #plt.savefig("grafico.png")


# Cargar los datos de entrenamiento y validación
train_data = load_data("lista_prueba.pkl")

# Separar entradas y salidas
input_data, output_data = map(list, zip(*train_data))

# Convertir a numpy arrays
X = np.array(input_data)
Y = np.array(output_data)


print(f"\nTotal de datos de entrada: {len(X)}\n")

# Dividir en 80% entrenamiento y 20% validación
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)     #random_state se fija para que la particion de los datos sea repetible

# Cargar el conjunto de prueba externo
test_data = load_data("lista_test.pkl")
X_test1, Y_test1 = map(list, zip(*test_data))

X_test = np.array(X_test1)
Y_test = np.array(Y_test1)

model = build_model()

model.summary()

# El parametro patience es la cantidad de epoch para comprobar la mejora
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

# Entrenar la red con validación
hist = model.fit(X_train, Y_train, 
                  epochs=1000, batch_size=32, 
                  validation_data=(X_val, Y_val), 
                  verbose=1, callbacks=[early_stop])

# Mostramos las graficas de MAE y MSE
plot_hist()

# Evaluar el modelo en el conjunto de prueba
test_loss, test_mae, test_mse = model.evaluate(X_test, Y_test, verbose=1)
print(f"MAE en test: {test_mae:5.4f} MPG")
print(f"MSE en test: {test_mse:5.4f} [$MPG^2$]")


# Guardar el modelo entrenado
model.save("WM_go2.keras")