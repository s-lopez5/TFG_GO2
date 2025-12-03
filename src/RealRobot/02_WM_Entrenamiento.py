import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Cargar los datos de entrenamiento
print("Cargando datos de entrenamiento...")
with open("training_data_shuffled.pkl", "rb") as f:
    data = pickle.load(f)
    inputs = data['inputs']
    outputs = data['outputs']

print(f"Forma de inputs: {inputs.shape}")
print(f"Forma de outputs: {outputs.shape}")

#Si outputs tiene forma (226, 1, 3), corregir a (226, 3)
if len(outputs.shape) == 3 and outputs.shape[1] == 1:
    outputs = outputs.squeeze(axis=1)
    print(f"Forma de outputs corregida: {outputs.shape}")

#Normalizar los datos
#Guardar estadísticas para desnormalizar después
input_mean = inputs.mean(axis=0)
input_std = inputs.std(axis=0)
output_mean = outputs.mean(axis=0)
output_std = outputs.std(axis=0)

#Evitar división por cero
input_std[input_std == 0] = 1
output_std[output_std == 0] = 1

inputs_normalized = (inputs - input_mean) / input_std
outputs_normalized = (outputs - output_mean) / output_std

print(f"\nEstadísticas de normalización:")
print(f"Input mean: {input_mean}")
print(f"Input std: {input_std}")
print(f"Output mean: {output_mean}")
print(f"Output std: {output_std}")

#Dividir en conjunto de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(
    inputs_normalized, outputs_normalized, 
    test_size=0.2, 
    random_state=42
)

print(f"\nConjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Conjunto de validación: {X_val.shape[0]} muestras")

#Construir el modelo de red neuronal
print("\nConstruyendo el modelo...")
model = keras.Sequential([
    layers.Input(shape=(4,)),  #[distanciaT, alfa_objT, alfa_robotT, action]
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),                    #Dropout (0.2) para prevenir overfitting
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(64, activation='relu'),
    
    layers.Dense(3)  # [distanciaT1, alfa_objT1, alfa_robotT1]
])

#Compilar el modelo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',         #Error cuadrático medio
    metrics=['mae']     #Mean absolute error
)

print("\nResumen del modelo:")
model.summary()

#Callbacks
#Early stopping para 50 epochs sin mejora
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True,
    verbose=1
)

#Reducir learning rate automáticamente
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    min_lr=1e-6,
    verbose=1
)


#Entrenar el modelo
print("\nEntrenando el modelo...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

#Guardar el modelo
print("\nGuardando el modelo...")
model.save("world_model.keras")

#Guardar las estadísticas de normalización
with open("normalization_stats.pkl", "wb") as f:
    pickle.dump({
        'input_mean': input_mean,
        'input_std': input_std,
        'output_mean': output_mean,
        'output_std': output_std
    }, f)

print("Modelo guardado como 'world_model.keras'")
print("Estadísticas guardadas como 'normalization_stats.pkl'")

#Evaluar el modelo
print("\nEvaluando el modelo...")
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)

print(f"Loss de entrenamiento: {train_loss:.6f}")
print(f"MAE de entrenamiento: {train_mae:.6f}")
print(f"Loss de validación: {val_loss:.6f}")
print(f"MAE de validación: {val_mae:.6f}")

#Hacer 50 predicciones de ejemplo
print("\n" + "="*80)
print("MOSTRANDO 50 PREDICCIONES DE EJEMPLO")
print("="*80)

num_ejemplos = min(50, len(X_val))  # Por si hay menos de 50 muestras en validación
ejemplos_inputs = X_val[:num_ejemplos]
ejemplos_outputs_real = y_val[:num_ejemplos]
ejemplos_outputs_pred = model.predict(ejemplos_inputs, verbose=0)

#Desnormalizar para visualizar
ejemplos_inputs_real = ejemplos_inputs * input_std + input_mean
ejemplos_outputs_real_desnorm = ejemplos_outputs_real * output_std + output_mean
ejemplos_outputs_pred_desnorm = ejemplos_outputs_pred * output_std + output_mean


for i in range(num_ejemplos):
    print(f"\n--- Ejemplo {i+1} ---")
    print(f"Input: dist={ejemplos_inputs_real[i][0]:.3f}, a_obj={ejemplos_inputs_real[i][1]:.3f}, a_robot={ejemplos_inputs_real[i][2]:.3f}, action={int(ejemplos_inputs_real[i][3])}")
    print(f"Real:  dist={ejemplos_outputs_real_desnorm[i][0]:.3f}, a_obj={ejemplos_outputs_real_desnorm[i][1]:.3f}, a_robot={ejemplos_outputs_real_desnorm[i][2]:.3f}")
    print(f"Pred:  dist={ejemplos_outputs_pred_desnorm[i][0]:.3f}, a_obj={ejemplos_outputs_pred_desnorm[i][1]:.3f}, a_robot={ejemplos_outputs_pred_desnorm[i][2]:.3f}")
    error = np.abs(ejemplos_outputs_real_desnorm[i] - ejemplos_outputs_pred_desnorm[i])
    print(f"Error: dist={error[0]:.3f}, a_obj={error[1]:.3f}, a_robot={error[2]:.3f}")

#Visualizar las 50 predicciones
fig = plt.figure(figsize=(16, 12))

variables = ['Distancia', 'Alfa Objetivo', 'Alfa Robot']
for idx, var_name in enumerate(variables):
    plt.subplot(3, 1, idx + 1)
    
    indices = np.arange(num_ejemplos)
    valores_reales = ejemplos_outputs_real_desnorm[:, idx]
    valores_predichos = ejemplos_outputs_pred_desnorm[:, idx]
    
    plt.plot(indices, valores_reales, 'o-', label='Real', color='blue', markersize=4, alpha=0.7)
    plt.plot(indices, valores_predichos, 's-', label='Predicho', color='red', markersize=4, alpha=0.7)
    
    plt.xlabel('Número de Ejemplo')
    plt.ylabel(var_name)
    plt.title(f'{var_name}: Valores Reales vs Predichos')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=150)
print("\n" + "="*80)
print("Gráfica de predicciones guardada como 'predictions_comparison.png'")
print("="*80)

#Visualizar el entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Época')
plt.ylabel('Loss (MSE)')
plt.title('Pérdida durante el entrenamiento')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.title('Error Absoluto Medio')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("\nGráfica guardada como 'training_history.png'")
plt.show()

#Análisis de errores por variable
print("\nAnálisis de errores por variable:")
all_predictions = model.predict(X_val, verbose=0)
all_predictions_desnorm = all_predictions * output_std + output_mean
y_val_desnorm = y_val * output_std + output_mean

errors = np.abs(all_predictions_desnorm - y_val_desnorm)
print(f"MAE Distancia: {errors[:, 0].mean():.4f}")
print(f"MAE Alfa Objetivo: {errors[:, 1].mean():.4f}")
print(f"MAE Alfa Robot: {errors[:, 2].mean():.4f}")
