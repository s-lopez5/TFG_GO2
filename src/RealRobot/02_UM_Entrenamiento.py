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

    with open("UM_utility_data_merged.pkl", "rb") as f:
        utility_data = pickle.load(f)
        inputs = utility_data['arrays']
        utilities = utility_data['utility']

    print(f"Datos cargados: {len(utility_data)} muestras")
    print(f"\nForma de inputs (percepciones en t+1): {inputs.shape}")
    print(f"Forma de utilidades: {utilities.shape}")

    return inputs, utilities

def plot_training_history(history):
    """
    Grafica y guarda las curvas de entrenamiento (MSE y MAE)
    
    Args:
        history: Objeto History de Keras con el historial de entrenamiento
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfica de MSE (Loss)
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('MSE (Mean Squared Error)', fontsize=12)
    ax1.set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Escala logarítmica para mejor visualización
    
    # Gráfica de MAE
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('MAE (Mean Absolute Error)', fontsize=12)
    ax2.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('UM_training_history_1.png', dpi=300, bbox_inches='tight')
    print("\nGráfica de entrenamiento guardada")
    plt.show()

def plot_prediction_examples(model, X_test, y_test, input_mean, input_std, n_examples=5):
    """
    Muestra ejemplos de predicciones vs valores reales
    
    Args:
        model: Modelo entrenado
        X_test: Datos de test normalizados
        y_test: Valores reales de utilidad
        input_mean: Media usada para normalización
        input_std: Desviación estándar usada para normalización
        n_examples: Número de ejemplos a mostrar
    """
    # Seleccionar ejemplos aleatorios
    indices = np.random.choice(len(X_test), n_examples, replace=False)
    
    # Hacer predicciones
    predictions = model.predict(X_test[indices], verbose=0)
    
    # Desnormalizar las percepciones para mostrarlas
    X_original = X_test[indices] * input_std + input_mean
    
    # Crear la figura
    fig, axes = plt.subplots(1, n_examples, figsize=(4*n_examples, 4))
    if n_examples == 1:
        axes = [axes]
    
    for i, (ax, idx) in enumerate(zip(axes, indices)):
        real_value = y_test[idx][0]
        pred_value = predictions[i][0]
        error = abs(real_value - pred_value)
        
        # Crear barra comparativa
        x_pos = [0, 1]
        values = [real_value, pred_value]
        colors = ['#2ecc71', '#3498db']
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Añadir valores en las barras
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Configurar el gráfico
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Real', 'Predicción'], fontsize=11)
        ax.set_ylabel('Utilidad', fontsize=11)
        ax.set_title(f'Ejemplo {i+1}\nError: {error:.3f}', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Añadir percepciones como texto
        percepciones_text = (
            f"Dist: {X_original[i][0]:.2f}m\n"
            f"a_obj: {X_original[i][1]:.2f}rad\n"
            f"a_robot: {X_original[i][2]:.2f}rad"
        )
        ax.text(0.5, 0.98, percepciones_text,
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Ejemplos de Predicciones del Modelo de Utilidad', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('UM_prediction_example_1.png', dpi=300, bbox_inches='tight')
    print("Gráfica de ejemplos guardada")
    plt.show()

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

    #Entrenar el modelo
    print("\n" + "="*60)
    print("ENTRENANDO MODELO")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    #Guardar el modelo
    print("\nGuardando el modelo...")
    model.save("utility_model_1.keras")

    #Evaluación
    print("\n" + "="*60)
    print("EVALUACIÓN")
    print("="*60 + "\n")
    
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Entrenamiento - MSE: {train_loss:.6f}, MAE: {train_mae:.6f}")
    print(f"Validación    - MSE: {val_loss:.6f}, MAE: {val_mae:.6f}")
    
    #Guardar el modelo final
    model.save("utility_model_1.keras")

    #Guardar parámetros de normalización
    normalization_params = {
        'input_mean': input_mean,
        'input_std': input_std
    }
    
    with open('normalization_params_UM.pkl', 'wb') as f:
        pickle.dump({
            'input_mean': input_mean,
            'input_std': input_std
        }, f)

    #Generar visualizaciones
    print("\n" + "="*60)
    print("GENERANDO VISUALIZACIONES")
    print("="*60 + "\n")
    
    #Gráfica de historial de entrenamiento
    plot_training_history(history)

    #Gráfica de ejemplos de predicción
    plot_prediction_examples(model, X_val, y_val, input_mean, input_std, n_examples=5)
    
    print("\n" + "="*60)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)