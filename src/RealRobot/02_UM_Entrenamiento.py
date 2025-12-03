import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_utility_data():
    """
    Carga los datos de utilidad desde un archivo pickle
    Formato esperado: lista de tuplas (percepciones_t1, utilidad)
    donde percepciones_t1 = [distancia_t1, alfa_obj_t1, alfa_robot_t1]
    
    Args:
        utility_data_path: Ruta al archivo con los datos de utilidad
    
    Returns:
        inputs: Array de percepciones en t+1 (distancia, alfa_obj, alfa_robot)
        utilities: Array de valores de utilidad
    """

    print("Cargando datos de utilidad...")

    with open("utility_data.pkl", "rb") as f:
        utility_data = pickle.load(f)

    print(f"Datos cargados: {len(utility_data)} muestras")

    #Extraer percepciones en t+1 y utilidades
    inputs = []
    utilities = []

    for item in utility_data:
        #item debería ser una tupla/lista: (percepciones_t1, utilidad)
        # percepciones_t1 = [distancia_t1, alfa_obj_t1, alfa_robot_t1]
        percepciones_t1 = item[0]  
        utilidad = item[1]          # Valor de utilidad asignado
        
        inputs.append(percepciones_t1)
        utilities.append(utilidad)

    inputs = np.array(inputs)
    utilities = np.array(utilities)

    print(f"\nForma de inputs (percepciones en t+1): {inputs.shape}")
    print(f"Forma de utilidades: {utilities.shape}")

    return inputs, utilities

if __name__ == "__main__":
    
    #Cargar dataset
    print("="*60)
    print("CARGANDO DATOS DE UTILIDAD")
    print("="*60 + "\n")
    
    try:
        X, Y = load_utility_data()
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo 'utility_data.pkl'")
        print("\nFormato esperado del archivo:")
        print("  - Lista de tuplas/listas")
        print("  - Cada elemento: ([distancia_t1, alfa_obj_t1, alfa_robot_t1], utilidad)")
        print("  - distancia_t1: Distancia al objetivo en t+1 (metros)")
        print("  - alfa_obj_t1: Ángulo hacia el objetivo en t+1 (radianes)")
        print("  - alfa_robot_t1: Orientación del robot en t+1 (radianes)")
        print("  - utilidad: Valor asignado a ese estado")
        print("\nEjemplo de estructura:")
        print("  [")
        print("    ([1.5, 0.785, 0.3], 0.2),")
        print("    ([1.2, 0.523, 0.1], 0.7),")
        print("    ([0.8, 0.314, -0.2], 1.0),")
        print("    ...")
        print("  ]")
        exit(1)
    
    #Verificar que tenemos 3 percepciones
    if X.shape[1] != 3:
        print(f"\nADVERTENCIA: Se esperaban 3 percepciones, pero se encontraron {X.shape[1]}")
        print("Formato esperado: [distancia_t1, alfa_obj_t1, alfa_robot_t1]")
    
    input_size = X.shape[1]
    print(f"\nTamaño de entrada: {input_size} valores (distancia, alfa_obj, alfa_robot)")
    
    #Normalizar los inputs (percepciones)
    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    
    #Evitar división por cero
    input_std[input_std == 0] = 1
    
    X_normalized = (X - input_mean) / input_std
    
    #No normalizamos las utilidades para mantener su significado
    y_values = Y.reshape(-1, 1)  # Reshape para que sea (n, 1)
    
    print(f"\nEstadísticas de normalización de percepciones:")
    print(f"  Media mínima: {input_mean.min():.3f}")
    print(f"  Media máxima: {input_mean.max():.3f}")
    print(f"  Std mínima: {input_std.min():.3f}")
    print(f"  Std máxima: {input_std.max():.3f}")
    
    #Dividir en conjunto de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_normalized, y_values,
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de validación: {X_val.shape[0]} muestras")


    #Construir el modelo de utilidad
    print("\n" + "="*60)
    print("CONSTRUYENDO MODELO DE UTILIDAD")
    print("="*60 + "\n")
    
    model = keras.Sequential([
        layers.Input(shape=(3,)),  # [distancia_t1, alfa_obj_t1, alfa_robot_t1]
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1)  # Valor de utilidad (escalar)
    ])
    
    #Compilar el modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("Resumen del modelo:")
    model.summary()
    
    #Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-6,
        verbose=1
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        'utility_model_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )