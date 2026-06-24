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

def random_action():
    """
    Devuelve la accion aleatoria continua en forma de array [x, y, z].
    x: velocidad en m/s (0.3 a 0.5)
    y: velocidad lateral en m/s (siempre 0)
    z: velocidad angular en rad/s (-1.047 a 1.047)
    """

    val_x = 0.0
    val_z = 0.0

    #0 = avanzar
    #1 = girar derecha
    #2 = girar izquierda

    action = random.randint(0, 2)

    if action == 0:
        val_x = random.choice([0.3, 0.4, 0.5])
    elif action == 1:
        val_z = random.choice([0.524, 0.785, 1.047])
    elif action == 2:
        val_z = random.choice([-1.047, -0.785, -0.524])

    return [val_x, 0.0, val_z]
    
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

    actual_pos = []         #Posicion actual del robot
    objetive_pos = []       #Posicion del objetivo
    finalizado = False      #Activar cuando el robot alcance el objetivo
    utility_data = []       #Lista de datos para el modelo de utilidad

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
        
        #Obtener una accion aleatoria
        action = random_action()  
        
        print(f"Acción seleccionada: {action}")

        #Ejecutar accion y esperar a que acabe el movimiento
        sport_client.Move(action[0], action[1], action[2])
        time.sleep(1)

        #Comprobar si el robot se ha salido de los límites establecidos
        if actual_pos[1][0] >= 2.0 or actual_pos[1][0] <= -1.1 or actual_pos[1][2] >= 1.25 or actual_pos[1][2] <= -1.4:
            
            out_of_limits()

            #Actualizar posicion despues del volver dentro de los limites
            actual_pos = streaming_client.get_last_pos()
            
            #Borrar todos los datos acumulados tras salir de los límites
            utility_data.clear()
            print("Datos borrados. Comenzando nueva recolección.\n")
            continue
        else:
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

            for i, elemento in enumerate(reversed(ultimos)):
                # Asignar valores de 1.0, 0.9, 0.8, ..., 0.0
                valor = 1.0 - (i * 0.1)
                datos_valorados.append((elemento, valor))
    
            # Invertir la lista para mantener el orden original
            datos_valorados.reverse()

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
    print(f"Datos valorados: {datos_valorados}")

    #Guardar los últimos 10 datos con sus valores
    with open("utility_data_with_models_1.pkl", "wb") as f:
        pickle.dump({
            'datos_objetivo': datos_valorados
        }, f)

    #Detener el cliente de streaming
    print("Deteniendo el cliente de streaming...")
    streaming_client.shutdown()

    # Cerrar la gráfica al finalizar
    if fig is not None:
        plt.close(fig)
