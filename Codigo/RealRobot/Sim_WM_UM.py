import numpy as np
import pickle
from dataclasses import dataclass
from tensorflow import keras

def calculate_best_action(world_model, utility_model, obs_t, input_mean_wm, input_std_wm, output_mean_wm, output_std_wm,
                         input_mean_um, input_std_um):
    """
    Calcula la mejor acción a ejecutar basándose en el modelo de mundo y el modelo de utilidad.
    
    Parámetros:
    - world_model: Modelo entrenado que predice las percepciones en T+1 dado el estado actual y una acción
    - utility_model: Modelo entrenado que predice el valor de utilidad de un estado
    - current_obs: Observaciones actuales [distancia, alfa_obj, alfa_robot]
    
    Retorna:
    - best_action: La acción con el mayor valor de utilidad predicho (0-5)
    """
    
    #Número de acciones posibles (0-5)
    num_actions = 6
    
    #Arrays para almacenar predicciones
    predicted_utilities = []
    
    #Para cada acción posible
    for action in range(num_actions):
        #Preparar la entrada para el modelo de mundo: [obs_actuales + acción]
        #current_obs tiene 3 valores: [distancia, alfa_obj, alfa_robot]
        world_model_input = np.concatenate([obs_t, [action]]).reshape(1, -1)

        #Normalizar la entrada
        world_model_input_normalized = (world_model_input - input_mean_wm) / input_std_wm
        
        #Predecir las observaciones en T+1 usando el modelo de mundo
        predicted_obs_t1_normalized_wm = world_model.predict(world_model_input_normalized, verbose=0)
        
        #Desnormalizar las observaciones predichas para visualización
        predicted_obs_t1 = (predicted_obs_t1_normalized_wm * output_std_wm) + output_mean_wm

        #Normalizar las observaciones predichas para el modelo de utilidad
        predicted_obs_t1_normalized_um = (predicted_obs_t1 - input_mean_um) / input_std_um
        
        """
        print("Formato:")
        print(f"Obs_t: {obs_t.shape}")
        print(f"Predicted Obs_t1 normalized WM: {predicted_obs_t1_normalized_wm.shape}")
        print(f"Predicted Obs_t1: {predicted_obs_t1.shape}")
        print(f"Predicted Obs_t1 normalized UM: {predicted_obs_t1_normalized_um.shape}\n\n")
        """

        #Usar el modelo de utilidad para predecir el valor de utilidad de las observaciones predichas
        predicted_utility = utility_model.predict(predicted_obs_t1_normalized_um, verbose=0)[0][0]
        
        print(f"Obs_t: {obs_t}, accion: {action}, obs_t1: {predicted_obs_t1.flatten()}, utilidad: {predicted_utility}")
        """
        print(f"Obs_t - Distancia: {obs_t[0]:.4f}, Alfa_obj: {obs_t[1]:.4f}, Alfa_robot: {obs_t[2]:.4f}")
        print(f"Action: {action}")
        predicted_obs_t1 = predicted_obs_t1.flatten()
        print(f"Predicted Obs_t1 - Distancia: {predicted_obs_t1[0]:.4f}, Alfa_obj: {predicted_obs_t1[1]:.4f}, Alfa_robot: {predicted_obs_t1[2]:.4f}")  
        print(f"Predicted Utility: {predicted_utility:.4f}\n\n")
        """

        predicted_utilities.append(predicted_utility)

    
    #Encontrar la acción con el mayor valor de utilidad
    best_action = np.argmax(predicted_utilities)
    best_utility = predicted_utilities[best_action]
    
    print(f"\n  Mejor acción seleccionada: {best_action} (Utilidad: {best_utility:.4f})")
    
    return best_action, predicted_obs_t1


def distancia(p1, p2):
    """
    Calcula la distancia entre el robot y el objetivo, tomando como referencia el punto medio 
    del objetivo y el marker delantero del robot.
    """
    return np.sqrt((p1[1][0] - p2[0])**2 + (p1[1][2] - p2[2])**2)

def alfa_obj(p1, p2):
    """
    Calcula el ángulo entre el robot y el objetivo, tomando como referencia el punto medio del objetivo 
    y el marker delantero del robot.
    """
    return np.arctan2(p1[1][2] - p2[2], p1[1][0] - p2[0])

def alfa_robot(p):
    """
    Calcula el ángulo de orientación del robot, tomando como referencia el marker delantero y trasero del robot.
    """
    return np.arctan2(p[1][2] - p[2][2], p[1][0] - p[2][0])


if __name__ == "__main__":

    #Cargar modelos entrenados
    print("Cargando modelos de mundo y utilidad...")
    world_model = keras.models.load_model("world_model_continuo_1168_2_ES50.keras")
    utility_model = keras.models.load_model("utility_model_1.keras")
    print("Modelos cargados correctamente.\n")

    #Cargar parámetros de normalización del modelo de mundo
    print("Cargando parámetros de normalización del modelo de mundo...")
    with open('normalization_stats_cont.pkl', 'rb') as f:
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

    actual_pos = []         #Posicion actual del robot
    objetive_pos = []       #Posicion del objetivo

    print("\nIniciando...\n")

    actual_pos = [[0.1, 0, 0.1], [0.1, 0, 0], [0, 0, 0]]
    objetive_pos = [0, 0, 0]

    #Calcular observaciones actuales
    distanciaT = distancia(actual_pos, objetive_pos)
    alfa_objT = alfa_obj(actual_pos, objetive_pos)
    alfa_robotT = alfa_robot(actual_pos)

    obs_t = np.array([distanciaT, alfa_objT, alfa_robotT])

    for i in range(1, 11):
        
        print(f"--- Iteración {i} ---")

        # Obtener la mejor acción usando los modelos
        action, predicted_obs_t1 = calculate_best_action(world_model, utility_model, obs_t,
                                    input_mean_wm, input_std_wm, output_mean_wm, output_std_wm,
                                    input_mean_um, input_std_um)
        
        obs_t = predicted_obs_t1.flatten()
        print(f"Posición actual actualizada del robot (simulada): {obs_t}\n")