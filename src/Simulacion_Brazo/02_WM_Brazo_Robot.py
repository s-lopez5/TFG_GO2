import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #Cargar los datos de entrenamiento
    print("Cargando datos de entrenamiento...")
    with open("WM_data_Brazo_Robot.pkl", "rb") as f:
        data = pickle.load(f)
        inputs = data['inputs']
        outputs = data['outputs']
    
    #Normalizar los datos
    #Guardar estadísticas para desnormalizar después
    input_mean = inputs.mean(axis=0)
    input_std = inputs.std(axis=0)
    output_mean = outputs.mean(axis=0)
    output_std = outputs.std(axis=0) 
    
    # Evitar división por cero en dimensiones constantes (ángulo sobre sí mismo)
    # Índice 2 en inputs y outputs
    input_std[2] = np.where(input_std[2] == 0, 1e-8, input_std[2])
    output_std[2] = np.where(output_std[2] == 0, 1e-8, output_std[2])

    inputs_normalized = (inputs - input_mean) / input_std
    outputs_normalized = (outputs - output_mean) / output_std

    #Dividir en conjunto de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        inputs_normalized, outputs_normalized,
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de validación: {X_val.shape[0]} muestras")

    #Construir el modelo de utilidad
    print("\n" + "="*60)
    print("CONSTRUYENDO MODELO DE UTILIDAD")
    print("="*60 + "\n")
    
    """
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

    model = keras.Sequential([
        layers.Input(shape=(3,)),   # [distancia_t1, alfa_obj_t1, alfa_robot_t1]
        
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.1),
        
        layers.Dense(1) # Valor de utilidad
    ])

    #Modelo simple
    model = keras.Sequential([
        layers.Input(shape=(3,)),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.1),
        
        layers.Dense(1)
    ])
    """

    model = keras.Sequential([
        layers.Input(shape=(3,)),   # [distancia_t1, alfa_obj_t1, alfa_robot_t1]
        
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.1),
        
        layers.Dense(1) # Valor de utilidad
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
        patience=100,
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

    #Entrenar el modelo
    print("\n" + "="*60)
    print("ENTRENANDO MODELO")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=32,
        callbacks=[reduce_lr],
        verbose=1
    )"""

    #Evaluación
    print("\n" + "="*60)
    print("EVALUACIÓN")
    print("="*60 + "\n")
    
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"Entrenamiento - MSE: {train_loss:.6f}, MAE: {train_mae:.6f}")
    print(f"Validación    - MSE: {val_loss:.6f}, MAE: {val_mae:.6f}")
    
    #==========================================
    #GRÁFICAS
    #==========================================
    
    #Gráfica 1: Evolución de MSE y MAE durante el entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    #MSE
    ax1.plot(history.history['loss'], label='Train MSE', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation MSE', linewidth=2)
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Evolución del MSE', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    #Límites para MSE
    ax1.set_xlim(0, len(history.history['loss']))
    mse_min = min(min(history.history['loss']), min(history.history['val_loss']))
    mse_max = max(max(history.history['loss']), max(history.history['val_loss']))
    ax1.set_ylim(0, mse_max * 1.1)
    
    #MAE
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Evolución del MAE', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Límites para MAE
    ax2.set_xlim(0, len(history.history['mae']))
    mae_min = min(min(history.history['mae']), min(history.history['val_mae']))
    mae_max = max(max(history.history['mae']), max(history.history['val_mae']))
    ax2.set_ylim(0, mae_max * 1.1)
    
    plt.tight_layout()
    plt.savefig('WM_Brazo_training_metrics.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfica de métricas guardada: training_metrics.png")
    plt.close()

    #Gráfica 2: Predicciones vs Valores Reales (solo distancia y ángulo al objetivo)
    #Hacer predicciones en el conjunto de validación
    y_pred_normalized = model.predict(X_val, verbose=0)
    
    #Desnormalizar para comparar en escala original
    y_pred = y_pred_normalized * output_std + output_mean
    y_val_original = y_val * output_std + output_mean
    
    #Crear figura con 2 subgráficas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    output_names = ['Distancia', 'Ángulo al Objetivo']
    output_indices = [0, 1]  # Solo índices 0 y 1
    
    for ax, name, idx in zip(axes, output_names, output_indices):
        ax.scatter(y_val_original[:, idx], y_pred[:, idx], alpha=0.5, s=20)
        
        #Línea diagonal perfecta (predicción = real)
        min_val = min(y_val_original[:, idx].min(), y_pred[:, idx].min())
        max_val = max(y_val_original[:, idx].max(), y_pred[:, idx].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
        
        ax.set_xlabel('Valores Reales', fontsize=12)
        ax.set_ylabel('Predicciones', fontsize=12)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        """
        # Establecer límites en los ejes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        """
        
        #Calcular R²
        correlation_matrix = np.corrcoef(y_val_original[:, idx], y_pred[:, idx])
        r_squared = correlation_matrix[0, 1]**2
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', 
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('WM_Brazo_Predicciones.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica de predicciones guardada: WM_Brazo_Predicciones.png")
    plt.close()

    #Guardar el modelo final
    model.save("utility_model_inverso.keras")

    #Guardar las estadísticas de normalización
    with open("normalization_stats_WM_Brazo.pkl", "wb") as f:
        pickle.dump({
            'input_mean': input_mean,
            'input_std': input_std,
            'output_mean': output_mean,
            'output_std': output_std
        }, f)