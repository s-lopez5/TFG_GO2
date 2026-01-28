import time
import sys
import os
import numpy as np
import random
import pickle
from dataclasses import dataclass
from tensorflow import keras

#Añadir el path del SDK de Unitree
from sdkpy.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from sdkpy.idl.default import unitree_go_msg_dds__SportModeState_
from sdkpy.idl.unitree_go.msg.dds_ import SportModeState_
from sdkpy.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)

#Añadir el path del SDK de Natnet
sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "NatNet_SDK", "samples", "PythonClient"))
if sdk_path not in sys.path:
    sys.path.append(sdk_path)

from NatNet_SDK.samples.PythonClient.NatNetClient import NatNetClient
from NatNet_SDK.samples.PythonClient.NatNetClient import NatNetClient
from NatNet_SDK.samples.PythonClient.DataDescriptions import DataDescriptions
from NatNet_SDK.samples.PythonClient.MoCapData import MoCapData

"""
def random_action():
    
    #Devuelve un número aleatorio entre 0 y 5 (ambos inclusive).
    #No devuelve el mismo número tres veces consecutivas.
    
    global action_history

    if len(action_history) >= 2 and action_history[-1] == action_history[-2]:
        # Si las dos últimas acciones son iguales, elegir una acción diferente
        action_rep = action_history[-1]
        action = random.randint(0, 5)
        while action == action_rep:
            action = random.randint(0, 5)
    else:
        action = random.randint(0, 5)
    
    #Agregar y mantener solo las dos últimas acciones en el historial
    action_history.append(action)
    if len(action_history) > 2:
        action_history.pop(0) 

    return action
    """

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

        #Usar el modelo de utilidad para predecir el valor de utilidad de las observaciones predichas
        predicted_utility = utility_model.predict(predicted_obs_t1_normalized_um, verbose=0)[0][0]
        
        print(f"Obs_t: {obs_t}, accion: {action}, obs_t1: {predicted_obs_t1.flatten()}, utilidad: {predicted_utility}")

        predicted_utilities.append(predicted_utility)
        
        print(f"  Acción {action}: Utilidad predicha = {predicted_utility:.4f}")
    
    #Encontrar la acción con el mayor valor de utilidad
    best_action = np.argmax(predicted_utilities)
    best_utility = predicted_utilities[best_action]
    
    print(f"\n  Mejor acción seleccionada: {best_action} (Utilidad: {best_utility:.4f})")
    
    return best_action


def out_of_limits():
    """
    Cuando el robot se sale de los límites establecidos, esta función lo detiene,
    lo hace retroceder, girar sobre si mismo 180 grados.
    """
    sport_client.StopMove()
    time.sleep(2)
    sport_client.Move(0,0,3.14159)
    time.sleep(3)
    sport_client.Move(0.7,0,0)
    time.sleep(3)

    return

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

def receive_new_frame_with_data(data_dict):
    order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount", #type: ignore  # noqa F841
                  "rigidBodyCount", "skeletonCount", "labeledMarkerCount",
                  "timecode", "timecodeSub", "timestamp", "isRecording",
                  "trackedModelsChanged", "offset", "mocap_data"]
    dump_args = True
    if dump_args is True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "= "
            if key in data_dict:
                out_string += str(data_dict[key]) + " "
            out_string += "/"
        #print(out_string)


if __name__ == "__main__":
    
    ChannelFactoryInitialize(0, "eno1")

    #Crear un diccionario con las opciones de conexión
    optionsDict = {}
    optionsDict["clientAddress"] = "192.168.123.149"
    optionsDict["serverAddress"] = "192.168.123.164"
    optionsDict["use_multicast"] = False
    optionsDict["stream_type"] = 'd'
    stream_type_arg = None

    #Crear el cliente NatNet
    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    #Configurar el cliente de streaming
    streaming_client.new_frame_with_data_listener = receive_new_frame_with_data

    #Iniciar el cliente de streaming
    #Se ejecuta de forma continua y en un hilo separado
    is_running = streaming_client.run(optionsDict["stream_type"])

    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    print("Esperando conexión...")
    time.sleep(2)  #Dar tiempo para conectar

    if streaming_client.connected() is False:
        print("ERROR: Could not connect properly.  Check that Motive streaming is on.") #type: ignore  # noqa F501
        try:
            sys.exit(2)
        except SystemExit:
            print("...")
        finally:
            print("exiting")
        
    sport_client = SportClient()  
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    #Cargar modelos entrenados
    print("Cargando modelos de mundo y utilidad...")
    world_model = keras.models.load_model("world_model_700_shuffled.keras")
    utility_model = keras.models.load_model("utility_model_best.keras")
    print("Modelos cargados correctamente.\n")

    #Cargar parámetros de normalización del modelo de mundo
    print("Cargando parámetros de normalización del modelo de mundo...")
    with open('normalization_stats_WM.pkl', 'rb') as f:
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
    utility_data = []       #Lista de datos para el modelo de utilidad

    print("\nIniciando...\n")
    sport_client.StandUp()
    time.sleep(2)
    sport_client.ClassicWalk(True)
    time.sleep(2)

    actual_pos = streaming_client.get_last_pos()
    objetive_pos = streaming_client.get_objetive_pos()

    for i in range(1, 501):
        
        print(f"--- Iteración {i} ---")

        #Calcular observaciones actuales
        distanciaT = distancia(actual_pos, objetive_pos)
        alfa_objT = alfa_obj(actual_pos, objetive_pos)
        alfa_robotT = alfa_robot(actual_pos)

        obs_t = np.array([distanciaT, alfa_objT, alfa_robotT])

        print(f"Observaciones actuales (T): distancia={distanciaT:.4f}, alfa_obj={alfa_objT:.4f}, alfa_robot={alfa_robotT:.4f}")
        
        # Obtener la mejor acción usando los modelos
        action = calculate_best_action(world_model, utility_model, obs_t,
                                    input_mean_wm, input_std_wm, output_mean_wm, output_std_wm,
                                    input_mean_um, input_std_um)
        
        print(f"Acción seleccionada: {action}")

        """
        45 grad = 0.785 rad
        60 grad = 1.047 rad
        90 grad = 1.57 rad
        """
        
        if action == 0:
            sport_client.Move(0.4,0,0)      #Avanzar
        elif action == 1:
            sport_client.Move(0.7,0,0)      #Avanzar más rápido
        elif action == 2:
            sport_client.Move(0,0,0.785)    #Girar 45 grados a la derecha
        elif action == 3:
            sport_client.Move(0,0,1.047)    #Girar 60 grados a la derecha
        elif action == 4:
            sport_client.Move(0,0,-0.785)    #Girar 45 grados a la izquierda    
        elif action == 5:
            sport_client.Move(0,0,-1.047)   #Girar 60 grados a la izquierda

        #Esperar a que acabe el movimiento
        time.sleep(3)
        
        #Comprobar si el robot se ha salido de los límites establecidos
        if actual_pos[1][0] >= 2.0 or actual_pos[1][0] <= -1.1 or actual_pos[1][2] >= 1.25 or actual_pos[1][2] <= -1.4:
            out_of_limits()
            actual_pos = streaming_client.get_last_pos()
            continue
        else:
            actual_pos = streaming_client.get_last_pos()

        #Obtenemos las observaciones en T+1
        obs_t1 = actual_pos  
        
        distanciaT1 = distancia(obs_t1, objetive_pos)
        alfa_objT1 = alfa_obj(obs_t1, objetive_pos)
        alfa_robotT1 = alfa_robot(obs_t1)

        print("\n")
        print(f"Posición actual(T+1):\n x={actual_pos[0][0]}, y={actual_pos[0][1]}, z={actual_pos[0][2]}\nx={actual_pos[1][0]}, y={actual_pos[1][1]}, z={actual_pos[1][2]}\nx={actual_pos[2][0]}, y={actual_pos[2][1]}, z={actual_pos[2][2]}")
        print(f"Observaciones en T+1: distancia={distanciaT1}, alfa_obj={alfa_objT1}, alfa_robot={alfa_robotT1}")
        print("\n")

        print("Distancia al objetivo:")
        print(distanciaT1)

        datos_t1 = np.array([distanciaT1, alfa_objT1, alfa_robotT1])
        utility_data.append(datos_t1)
        
        #Comprobar si el robot ha alcanzado el objetivo
        if distanciaT1 <= 0.3:
            print("Objetivo alcanzado.\n")
            
            sport_client.StopMove()
            time.sleep(2)

            #Obtener los últimos 10 datos y asignar valores de 0.0 a 1.0
            ultimos = utility_data[-10:] if len(utility_data) >= 10 else utility_data
            datos_valorados = []

            for i, dato in enumerate(ultimos):
                #Calcular valor ascendente de 0.0 a 1.0
                valor = i / (len(ultimos) - 1) if len(ultimos) > 1 else 1.0
                datos_valorados.append((dato, valor))

            print(f"\nÚltimos {len(ultimos)} datos con valores asignados:")
            for idx, (dato, valor) in enumerate(datos_valorados):
                print(f"Dato {idx+1}: {dato} - Valor: {valor:.2f}")

            break
            
    sport_client.StopMove()
    time.sleep(2)
    sport_client.StandDown()
    time.sleep(3)

    print(f"Total de transiciones recolectadas: {len(utility_data)}")
    print(f"Transiciones recolectadas: {utility_data}")

    #Guardar los últimos 10 datos con sus valores
    with open("utility_data_with_models_1.pkl", "wb") as f:
        pickle.dump({
            'datos_objetivo': datos_valorados
        }, f)

    #Detener el cliente de streaming
    print("Deteniendo el cliente de streaming...")
    streaming_client.shutdown()