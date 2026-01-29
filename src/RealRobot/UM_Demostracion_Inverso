import time
import sys
import os
import numpy as np
import random
import pickle
from dataclasses import dataclass
from tensorflow import keras
import matplotlib.pyplot as plt

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

# Configurar matplotlib para modo interactivo
plt.ion()

# Variable global para la figura
fig = None
ax = None    

def calculate_best_action(world_model, utility_model, obs_t, input_mean_wm, input_std_wm, output_mean_wm, output_std_wm,
                         input_mean_um, input_std_um, iteration):

    #Lista de acciones posibles
    action_list = [[0.3, 0.0, 0.0], [0.4, 0.0, 0.0], [0.5, 0.0, 0.0], 
                   [0.0, 0.0, 0.524], [0.0, 0.0, 0.785], [0.0, 0.0, 1.047], 
                   [0.0, 0.0, -0.524], [0.0, 0.0, -0.785], [0.0, 0.0, -1.047]]
    
    #Arrays para almacenar predicciones
    predicted_utilities = []

    for action in action_list:

        #Preparar la entrada para el modelo de mundo: [obs_actuales + acción]
        #current_obs tiene 3 valores: [distancia, alfa_obj, alfa_robot]
        world_model_input = np.concatenate([obs_t, action]).reshape(1, -1)

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
        
        predicted_utilities.append([action, predicted_obs_t1, predicted_utility])

    #Encontrar la acción con el mayor valor de utilidad
    best_action, _, best_utility = max(predicted_utilities, key=lambda x: x[2])

    print(f"\n  Mejor acción seleccionada: {best_action} (Utilidad: {best_utility:.4f})")
    
    #Mostrar la gráfica con la acción seleccionada
    plot_utilities(predicted_utilities, best_action, iteration)

    return best_action
    
def plot_utilities(predicted_utilities, best_action, iteration):
    """
    Actualiza la gráfica con la utilidad de la acción seleccionada.
    La gráfica permanece abierta y se actualiza en cada iteración.
    """
    global fig, ax
    
    # Encontrar la utilidad de la mejor acción
    best_utility = None
    for action, _, utility in predicted_utilities:
        if action == best_action:
            best_utility = utility
            break
    
    # Crear etiqueta descriptiva para la acción seleccionada
    if best_action[0] > 0:
        label = f"Avanzar\n{best_action[0]:.1f} m/s"
    elif best_action[2] > 0:
        label = f"Girar Derecha\n{best_action[2]:.3f} rad/s"
    elif best_action[2] < 0:
        label = f"Girar Izquierda\n{abs(best_action[2]):.3f} rad/s"
    else:
        label = "Parar"
    
    # Crear la figura solo la primera vez
    if fig is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.canvas.manager.set_window_title('Utilidad de la Accion')
    
    # Limpiar el contenido anterior
    ax.clear()
    
    # Crear la barra
    bar = ax.bar([0], [best_utility], color='green', alpha=0.7, edgecolor='black', width=0.5)
    
    # Añadir valor encima de la barra
    ax.text(0, best_utility, f'{best_utility:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Configurar la gráfica
    ax.set_xlabel('Acción Seleccionada', fontsize=12, fontweight='bold')
    ax.set_ylabel('Utilidad Predicha', fontsize=12, fontweight='bold')
    ax.set_title(f'Iteración {iteration} - Utilidad de la Acción Seleccionada\n{label}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks([0])
    ax.set_xticklabels([label], fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(best_utility * 1.2, 1.0))
    
    # Actualizar la ventana
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)  # Pausa breve para permitir la actualización

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
    optionsDict["clientAddress"] = "192.168.0.138"
    optionsDict["serverAddress"] = "192.168.0.157"
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
    world_model = keras.models.load_model("world_model_continuo_1168_2_ES50.keras")
    utility_model = keras.models.load_model("utility_model_inverso.keras")
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
    with open('normalization_params_inverso_UM.pkl', 'rb') as f:
        norm_params_um = pickle.load(f)
        input_mean_um = norm_params_um['input_mean']
        input_std_um = norm_params_um['input_std']
    print("Parámetros de normalización del modelo de utilidad cargados.\n")

    actual_pos = []         #Posicion actual del robot
    objetive_pos = []       #Posicion del objetivo
    finalizado = False      #Activar cuando el robot alcance el objetivo

    print("\nIniciando...\n")
    sport_client.StandUp()
    time.sleep(2)
    sport_client.ClassicWalk(True)
    time.sleep(2)

    actual_pos = streaming_client.get_last_pos()

    for i in range(1, 501):
        
        print(f"--- Iteración {i} ---")

        objetive_pos = streaming_client.get_objetive_pos()  #Actualizar la posición del objetivo

        #Calcular observaciones actuales
        distanciaT = distancia(actual_pos, objetive_pos)
        alfa_objT = alfa_obj(actual_pos, objetive_pos)
        alfa_robotT = alfa_robot(actual_pos)

        obs_t = np.array([distanciaT, alfa_objT, alfa_robotT])

        print(f"Observaciones actuales (T): distancia={distanciaT:.4f}, alfa_obj={alfa_objT:.4f}, alfa_robot={alfa_robotT:.4f}")
        
        # Obtener la mejor acción usando los modelos
        action = calculate_best_action(world_model, utility_model, obs_t,
                                    input_mean_wm, input_std_wm, output_mean_wm, output_std_wm,
                                    input_mean_um, input_std_um, i)
        
        print(f"Acción seleccionada: {action}")

        #Ejecutar accion y esperar a que acabe el movimiento
        sport_client.Move(action[0], action[1], action[2])
        time.sleep(1)

        #Actualizar la posición actual del robot
        actual_pos = streaming_client.get_last_pos()

        #Obtenemos las observaciones en T+1
        distanciaT1 = distancia(actual_pos, objetive_pos)
        alfa_objT1 = alfa_obj(actual_pos, objetive_pos)
        alfa_robotT1 = alfa_robot(actual_pos)

        print("\n")
        print(f"Posición actual(T+1):\n x={actual_pos[0][0]}, y={actual_pos[0][1]}, z={actual_pos[0][2]}\nx={actual_pos[1][0]}, y={actual_pos[1][1]}, z={actual_pos[1][2]}\nx={actual_pos[2][0]}, y={actual_pos[2][1]}, z={actual_pos[2][2]}")
        print(f"Observaciones en T+1: distancia={distanciaT1}, alfa_obj={alfa_objT1}, alfa_robot={alfa_robotT1}")
        print("\n")

        print("Distancia al objetivo:")
        print(distanciaT1)

        #Comprobar si el robot se ha salido de los límites establecidos
        if actual_pos[1][0] >= 2.0 or actual_pos[1][0] <= -1.1 or actual_pos[1][2] >= 1.25 or actual_pos[1][2] <= -1.4:
            print("Objetivo alcanzado.\n")
            
            sport_client.StopMove()
            time.sleep(2)

            break

    sport_client.StopMove()
    time.sleep(2)
    sport_client.StandDown()
    time.sleep(3)

    #Detener el cliente de streaming
    print("Deteniendo el cliente de streaming...")
    streaming_client.shutdown()

    # Cerrar la gráfica al finalizar
    if fig is not None:
        plt.close(fig)
