import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("="*80)
print("ENTRENAMIENTO DEL MODELO DE UTILIDAD")
print("="*80)

#Cargar el modelo de mundo entrenado
print("\nCargando modelo de mundo...")
world_model = keras.models.load_model("world_model.keras")
print("✓ Modelo de mundo cargado")

#Cargar las estadísticas de normalización
print("Cargando estadísticas de normalización...")
with open("normalization_stats.pkl", "rb") as f:
    norm_stats = pickle.load(f)
    input_mean = norm_stats['input_mean']
    input_std = norm_stats['input_std']
    output_mean = norm_stats['output_mean']
    output_std = norm_stats['output_std']
print("✓ Estadísticas cargadas")

#Parámetros de simulación
NUM_EPISODES = 5000  # Número de episodios de entrenamiento
MAX_STEPS = 50      # Máximo de pasos por episodio
NUM_ACTIONS = 3     # 0: girar izq, 1: avanzar, 2: girar der

#Función para calcular la recompensa
def calculate_reward(state, next_state, step):
    """
    Calcula la recompensa basada en:
    - Reducción de distancia al objetivo (positivo)
    - Penalización por cada paso (eficiencia)
    - Bonus por alcanzar el objetivo
    """
    distancia_actual = state[0]
    distancia_siguiente = next_state[0]
    
    # Recompensa por reducir distancia
    reward = (distancia_actual - distancia_siguiente) * 10.0
    
    # Penalización por paso (fomenta eficiencia)
    reward -= 0.1
    
    # Bonus grande si alcanza el objetivo (distancia < 0.1)
    if distancia_siguiente < 0.1:
        reward += 100.0
    
    # Penalización si se aleja demasiado
    if distancia_siguiente > distancia_actual + 0.5:
        reward -= 5.0
    
    return reward

#Función para verificar si se alcanzó el objetivo
def reached_goal(state, threshold=0.1):
    return state[0] < threshold

#Función para simular usando el modelo de mundo
def simulate_step(state, action):
    """
    Usa el modelo de mundo para predecir el siguiente estado
    state: [distancia, alfa_obj, alfa_robot]
    action: 0, 1, o 2
    """
    # Preparar input: [distancia, alfa_obj, alfa_robot, action]
    model_input = np.array([[state[0], state[1], state[2], action]])
    
    # Normalizar
    model_input_norm = (model_input - input_mean) / input_std
    
    # Predecir siguiente estado
    next_state_norm = world_model.predict(model_input_norm, verbose=0)[0]
    
    # Desnormalizar
    next_state = next_state_norm * output_std + output_mean
    
    return next_state

#Función para generar un estado inicial aleatorio
def generate_initial_state():
    """Genera un estado inicial aleatorio"""
    distancia = np.random.uniform(0.5, 3.0)  # Distancia entre 0.5 y 3.0
    alfa_obj = np.random.uniform(-np.pi, np.pi)  # Ángulo objeto
    alfa_robot = np.random.uniform(-np.pi, np.pi)  # Ángulo robot
    return np.array([distancia, alfa_obj, alfa_robot])

#Generar datos de entrenamiento usando el modelo de mundo
print("\nGenerando datos de entrenamiento mediante simulación...")
training_states = []
training_actions = []
training_q_values = []

#Estrategia epsilon-greedy para exploración
epsilon = 1.0  # Probabilidad inicial de acción aleatoria
epsilon_decay = 0.995
epsilon_min = 0.1

episode_rewards = []
episode_steps = []

for episode in range(NUM_EPISODES):
    state = generate_initial_state()
    episode_reward = 0
    episode_data = []  # [(state, action, reward), ...]
    
    for step in range(MAX_STEPS):
        # Epsilon-greedy: exploración vs explotación
        if np.random.random() < epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            # Evaluar todas las acciones posibles
            q_values = []
            for a in range(NUM_ACTIONS):
                next_state = simulate_step(state, a)
                reward = calculate_reward(state, next_state, step)
                q_values.append(reward)
            action = np.argmax(q_values)
        
        # Simular la acción
        next_state = simulate_step(state, action)
        reward = calculate_reward(state, next_state, step)
        
        # Guardar transición
        episode_data.append((state.copy(), action, reward, next_state.copy()))
        
        episode_reward += reward
        
        # Verificar si alcanzó el objetivo
        if reached_goal(next_state):
            # Bonus por alcanzar el objetivo
            reward += 100.0
            episode_data[-1] = (state.copy(), action, reward, next_state.copy())
            break
        
        state = next_state
    
    # Calcular Q-values con retorno descontado (discount factor = 0.95)
    gamma = 0.95
    G = 0  # Retorno acumulado
    for i in reversed(range(len(episode_data))):
        state_i, action_i, reward_i, next_state_i = episode_data[i]
        G = reward_i + gamma * G
        
        training_states.append(state_i)
        training_actions.append(action_i)
        training_q_values.append(G)
    
    episode_rewards.append(episode_reward)
    episode_steps.append(len(episode_data))
    
    # Reducir epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if (episode + 1) % 500 == 0:
        avg_reward = np.mean(episode_rewards[-500:])
        avg_steps = np.mean(episode_steps[-500:])
        print(f"Episodio {episode + 1}/{NUM_EPISODES} | "
              f"Reward promedio: {avg_reward:.2f} | "
              f"Pasos promedio: {avg_steps:.1f} | "
              f"Epsilon: {epsilon:.3f}")

#Convertir a arrays
X_states = np.array(training_states)
y_actions = np.array(training_actions)
y_q_values = np.array(training_q_values)

print(f"\n✓ Datos generados: {len(X_states)} transiciones")
print(f"Forma de estados: {X_states.shape}")

#Preparar datos para el modelo de utilidad
#El modelo predecirá Q-values para cada acción dado un estado
#Crear matriz de Q-values: para cada muestra, solo la acción tomada tiene el Q-value calculado
X_train_states, X_val_states, y_train_actions, y_val_actions, y_train_q, y_val_q = train_test_split(
    X_states, y_actions, y_q_values, test_size=0.2, random_state=42
)

print(f"\nConjunto de entrenamiento: {X_train_states.shape[0]} muestras")
print(f"Conjunto de validación: {X_val_states.shape[0]} muestras")

#Construir el modelo de utilidad
print("\nConstruyendo modelo de utilidad...")
utility_model = keras.Sequential([
    layers.Input(shape=(3,)),  # [distancia, alfa_obj, alfa_robot]
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(64, activation='relu'),
    
    layers.Dense(NUM_ACTIONS)  # Q-values para cada acción
])

#Compilar el modelo
utility_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\nResumen del modelo de utilidad:")
utility_model.summary()

#Preparar los targets: matriz de Q-values
#Solo entrenamos el Q-value de la acción tomada
y_train_q_matrix = np.zeros((len(X_train_states), NUM_ACTIONS))
y_val_q_matrix = np.zeros((len(X_val_states), NUM_ACTIONS))

for i in range(len(X_train_states)):
    y_train_q_matrix[i, y_train_actions[i]] = y_train_q[i]

for i in range(len(X_val_states)):
    y_val_q_matrix[i, y_val_actions[i]] = y_val_q[i]

#Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

#Entrenar el modelo
print("\nEntrenando modelo de utilidad...")
history = utility_model.fit(
    X_train_states, y_train_q_matrix,
    validation_data=(X_val_states, y_val_q_matrix),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

#Guardar el modelo de utilidad
print("\nGuardando modelo de utilidad...")
utility_model.save("utility_model.keras")
print("✓ Modelo guardado como 'utility_model.keras'")

#Evaluar el modelo
print("\nEvaluando modelo de utilidad...")
train_loss, train_mae = utility_model.evaluate(X_train_states, y_train_q_matrix, verbose=0)
val_loss, val_mae = utility_model.evaluate(X_val_states, y_val_q_matrix, verbose=0)

print(f"Loss de entrenamiento: {train_loss:.6f}")
print(f"MAE de entrenamiento: {train_mae:.6f}")
print(f"Loss de validación: {val_loss:.6f}")
print(f"MAE de validación: {val_mae:.6f}")

#Visualizaciones
fig = plt.figure(figsize=(16, 10))

#Gráfica 1: Recompensas por episodio
plt.subplot(2, 3, 1)
plt.plot(episode_rewards, alpha=0.3, color='blue')
window = 100
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(moving_avg, color='red', linewidth=2, label=f'Media móvil ({window})')
plt.xlabel('Episodio')
plt.ylabel('Recompensa Total')
plt.title('Recompensa por Episodio')
plt.legend()
plt.grid(True, alpha=0.3)

#Gráfica 2: Pasos por episodio
plt.subplot(2, 3, 2)
plt.plot(episode_steps, alpha=0.3, color='green')
moving_avg_steps = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
plt.plot(moving_avg_steps, color='red', linewidth=2, label=f'Media móvil ({window})')
plt.xlabel('Episodio')
plt.ylabel('Número de Pasos')
plt.title('Pasos por Episodio')
plt.legend()
plt.grid(True, alpha=0.3)

#Gráfica 3: Loss del modelo de utilidad
plt.subplot(2, 3, 3)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Época')
plt.ylabel('Loss (MSE)')
plt.title('Pérdida del Modelo de Utilidad')
plt.legend()
plt.grid(True, alpha=0.3)

#Gráfica 4: MAE del modelo de utilidad
plt.subplot(2, 3, 4)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.title('Error Absoluto Medio')
plt.legend()
plt.grid(True, alpha=0.3)

#Gráfica 5: Distribución de acciones tomadas
plt.subplot(2, 3, 5)
action_names = ['Girar Izq', 'Avanzar', 'Girar Der']
action_counts = [np.sum(y_actions == i) for i in range(NUM_ACTIONS)]
plt.bar(action_names, action_counts, color=['blue', 'green', 'red'])
plt.xlabel('Acción')
plt.ylabel('Frecuencia')
plt.title('Distribución de Acciones')
plt.grid(True, alpha=0.3, axis='y')

#Gráfica 6: Predicción de ejemplo
plt.subplot(2, 3, 6)
example_state = X_val_states[0:1]
predicted_q_values = utility_model.predict(example_state, verbose=0)[0]
plt.bar(action_names, predicted_q_values, color=['blue', 'green', 'red'])
plt.xlabel('Acción')
plt.ylabel('Q-Value')
plt.title(f'Ejemplo de Q-Values Predichos\nEstado: dist={example_state[0][0]:.2f}, α_obj={example_state[0][1]:.2f}, α_robot={example_state[0][2]:.2f}')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('utility_model_training.png', dpi=150)
print("\n✓ Gráficas guardadas como 'utility_model_training.png'")
plt.show()

print("\n" + "="*80)
print("ENTRENAMIENTO COMPLETADO")
print("="*80)
print(f"Modelo de utilidad entrenado con {len(X_states)} transiciones")
print(f"Recompensa promedio final: {np.mean(episode_rewards[-100:]):.2f}")
print(f"Pasos promedio final: {np.mean(episode_steps[-100:]):.1f}")
print("="*80)