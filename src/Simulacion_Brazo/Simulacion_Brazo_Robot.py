import numpy as np
import random
import pickle
from dataclasses import dataclass
from tensorflow import keras

def distancia(x_prima, y_prima, x_2, y_2):
    """
    Calcula la distancia entre el punto (x_prima, y_prima) y el punto (x_2, y_2).
    """
    return np.sqrt((x_2 - x_prima)**2 + (y_2 - y_prima)**2)

def alfa_obj(x_prima, y_prima, x_2, y_2):
    """
    Calcula el ángulo entre el punto (x_prima, y_prima) y el punto (x_2, y_2).
    """
    return np.arctan2(y_2 - y_prima, x_2 - x_prima)

def pos_x2(L, alfa, beta):
    """
    Calcula la posición x del punto robot en función de la longitud L y los ángulos alfa y beta.
    """
    return L * (np.cos(np.radians(alfa)) + np.cos(np.radians(beta)))

def pos_y2(L, alfa, beta):
    """
    Calcula la posición y del punto robot en función de la longitud L y los ángulos alfa y beta.
    """
    return L * (np.sin(np.radians(alfa)) + np.sin(np.radians(beta)))

def pos_objetivo(L):
    """
    Devuelve una posición aleatoria para el objetivo dentro de los límites establecidos.
    """
    
    alfa, beta = random.randint(0, 90), random.randint(0, 90)
    x_prima = L * (np.cos(np.radians(alfa)) + np.cos(np.radians(beta)))
    y_prima = L * (np.sin(np.radians(alfa)) + np.sin(np.radians(beta)))
    return x_prima, y_prima
    
def calculate_best_action(world_model, utility_model, obs_t, input_mean_wm, input_std_wm, output_mean_wm, output_std_wm,
                         input_mean_um, input_std_um):
    """
    Calcula la mejor acción a ejecutar basándose en el modelo de mundo y el modelo de utilidad.
    
    Parámetros:
    - world_model: Modelo entrenado que predice las percepciones en T+1 dado el estado actual y una acción
    - utility_model: Modelo entrenado que predice el valor de utilidad de un estado
    - obs_t: Observaciones actuales [distancia, alfa_obj, 0, alfa_robot, beta_robot]
    - input_mean_wm, input_std_wm: Parámetros de normalización de entrada del modelo de mundo
    - output_mean_wm, output_std_wm: Parámetros de normalización de salida del modelo de mundo
    - input_mean_um, input_std_um: Parámetros de normalización del modelo de utilidad
    
    Retorna:
    - best_action: Tupla (delta_alfa, delta_beta) con la mejor acción
    - predicted_obs_t1: Observación predicha en t+1 tras ejecutar la mejor acción
    """

    best_utility = -np.inf
    best_action = None
    predicted_obs_t1 = None
    
    #Generar un conjunto de acciones posibles
    possible_actions = []
    for delta_alfa in range(-15, 16, 5):  # De -15 a 15 en pasos de 5
        for delta_beta in range(-15, 16, 5):
            possible_actions.append((delta_alfa, delta_beta))

    #Evaluar cada acción posible
    for action in possible_actions:
        delta_alfa, delta_beta = action
        
        #Crear el input para el modelo de mundo: [obs_t + acción]
        #obs_t tiene forma [distancia, alfa_obj, 0]
        wm_input = np.concatenate([obs_t.flatten(), [delta_alfa, delta_beta]])
        wm_input = wm_input.reshape(1, -1)
        
        #Normalizar la entrada del modelo de mundo
        wm_input_normalized = (wm_input - input_mean_wm) / (input_std_wm + 1e-8)
        
        #Predecir el siguiente estado con el modelo de mundo
        predicted_state_normalized = world_model.predict(wm_input_normalized, verbose=0)
        
        #Desnormalizar la salida del modelo de mundo
        predicted_state = predicted_state_normalized * output_std_wm + output_mean_wm
        
        #Normalizar la entrada para el modelo de utilidad
        um_input_normalized = (predicted_state - input_mean_um) / (input_std_um + 1e-8)
        
        #Predecir la utilidad del estado predicho
        utility = utility_model.predict(um_input_normalized, verbose=0)[0, 0]
        
        print(f"Acción: Δα={delta_alfa}°, Δβ={delta_beta}° | Predicted State: {predicted_state[0][0]:.4f}, {predicted_state[0][1]:.2f} | Predicted Utility: {utility:.4f}")

        #Actualizar la mejor acción si esta tiene mayor utilidad
        if utility > best_utility:
            best_utility = utility
            best_action = action
            predicted_obs_t1 = predicted_state
    
    print(f"Mejor acción: Δα={best_action[0]}°, Δβ={best_action[1]}° | Utilidad esperada: {best_utility:.4f}")
    
    return best_action, predicted_obs_t1       

def random_action():
    """
    Genera una acción aleatoria para los ángulos alfa y beta.
    Las acciones son incrementos de entre -15 y 15 grados.
    """

    return random.randint(-15, 15), random.randint(-15, 15)

if __name__ == "__main__":

    #Cargar modelos entrenados
    print("Cargando modelos de mundo y utilidad...")
    world_model = keras.models.load_model("WM_brazo_robot.keras")
    utility_model = keras.models.load_model("utility_model_100_1.keras")
    print("Modelos cargados correctamente.\n")

    #Cargar parámetros de normalización del modelo de mundo
    print("Cargando parámetros de normalización del modelo de mundo...")
    with open('normalization_stats_WM_Brazo.pkl', 'rb') as f:
        norm_params_wm = pickle.load(f)
        input_mean_wm = norm_params_wm['input_mean']
        input_std_wm = norm_params_wm['input_std']
        output_mean_wm = norm_params_wm['output_mean']
        output_std_wm = norm_params_wm['output_std']
    print("Parámetros de normalización del modelo de mundo cargados.\n")

    #Cargar parámetros de normalización del modelo de utilidad
    print("Cargando parámetros de normalización del modelo de utilidad...")
    with open('normalization_params_UM.pkl', 'rb') as f:
        norm_params_um = pickle.load(f)
        input_mean_um = norm_params_um['input_mean']
        input_std_um = norm_params_um['input_std']
    print("Parámetros de normalización del modelo de utilidad cargados.\n")
    
    L = 1.0  #Longitud del brazo del robot

    #Posicion del objetivo inicial
    x_obj, y_obj = pos_objetivo(L)
    print(f"Posición objetivo: ({x_obj}, {y_obj})")

    #Ángulos iniciales
    alfa, beta = random.randint(0, 90), random.randint(0, 90)  

    #Posicion en el paso t
    x_robot = pos_x2(L, alfa, beta)
    y_robot = pos_y2(L, alfa, beta)

    dist = distancia(x_obj, y_obj, x_robot, y_robot)
    angle_to_obj = alfa_obj(x_obj, y_obj, x_robot, y_robot)

    #Observación inicial
    obs_t = np.array([dist, angle_to_obj, 0])  #[distancia, alfa_obj, 0]

    for i in range(1, 6):
        
        print(f"--- Iteración {i} ---")

        print(f"Observación actual del robot (simulada): Distancia={obs_t[0]:.4f}, Alfa_obj={obs_t[1]:.4f}, 0")

        #Obtener la mejor acción usando los modelos
        action, predicted_obs_t1 = calculate_best_action(world_model, utility_model, obs_t,
                                    input_mean_wm, input_std_wm, output_mean_wm, output_std_wm,
                                    input_mean_um, input_std_um)

        obs_t = predicted_obs_t1.flatten()
        print(f"Obs_t1 actualizada del robot (simulada): {obs_t[0]:.4f}, {obs_t[1]:.4f}, {obs_t[2]:.4f}\n")

