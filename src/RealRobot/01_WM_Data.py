import time
import sys
import os
import numpy as np
import random
import pickle
import select
import termios
import tty
from dataclasses import dataclass

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


@dataclass
class TestOption:
    name: str
    id: int

option_list = [
    TestOption(name="stand_up", id=0),         
    TestOption(name="stand_down", id=1),     
    TestOption(name="move forward 1", id=2),   
    TestOption(name="move forward 2", id=3),         
    TestOption(name="rotate (45grad)", id=4),    
    TestOption(name="rotate (60grad)", id=5),
    TestOption(name="rotate (-45grad)", id=6),
    TestOption(name="rotate (-60grad)", id=7),  
    TestOption(name="stop_move", id=8)       
]

action_history = []  #Historial de acciones tomadas

class UserInterface:
    def __init__(self):
        self.test_option_ = None

    def convert_to_int(self, input_str):
        try:
            return int(input_str)
        except ValueError:
            return None

    def terminal_handle(self):
        
        input_str = input("\nEnter id (or list): \n")

        if input_str == "list":
            self.test_option_.name = None
            self.test_option_.id = None
            for option in option_list:
                print(f"{option.name}, id: {option.id}")
            return

        for option in option_list:
            if input_str == option.name or self.convert_to_int(input_str) == option.id:
                self.test_option_.name = option.name
                self.test_option_.id = option.id
                print(f"Test: {self.test_option_.name}, test_id: {self.test_option_.id}")
                return

        print("No matching test option found.")


def is_esc_pressed():
    """
    Detecta si se ha presionado la tecla ESC de forma no bloqueante.
    Retorna True si se presionó ESC, False en caso contrario.
    """
    # Guardar configuración original del terminal
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        # Configurar terminal en modo raw para detectar teclas
        tty.setcbreak(sys.stdin.fileno())
        
        # Verificar si hay entrada disponible (sin bloquear)
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            # ESC tiene código ASCII 27
            if ord(key) == 27:
                return True
    finally:
        # Restaurar configuración original del terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    return False

def random_action():
    """
    Devuelve un número aleatorio entre 0 y 5 (ambos inclusive).
    No devuelve el mismo número tres veces consecutivas.
    """
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
    return np.sqrt((p1[0][0] - p2[0])**2 + (p1[0][2] - p2[2])**2)

def alfa_obj(p1, p2):
    """
    Calcula el ángulo entre el robot y el objetivo, tomando como referencia el punto medio del objetivo 
    y el marker delantero del robot.
    """
    return np.arctan2(p1[0][2] - p2[2], p1[0][0] - p2[0])

def alfa_robot(p):
    """
    Calcula el ángulo de orientación del robot, tomando como referencia el marker delantero y trasero del robot.
    """
    return np.arctan2(p[0][2] - p[1][2], p[0][0] - p[1][0])

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
    optionsDict["serverAddress"] = "192.168.123.112"
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
        
    test_option = TestOption(name=None, id=None) 
    user_interface = UserInterface()
    user_interface.test_option_ = test_option

    sport_client = SportClient()  
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    actual_pos = []         #Posicion actual del robot
    objetive_pos = []       #Posicion del objetivo
    trainnig_data = []      #Lista de datos de entrenamiento
    trainnig_data_2 = []    #Lista de datos de entrenamiento con posiciones absolutas

    print("Presiona ESC para salir del bucle y guardar datos...")
    print("-" * 50)

    print("\nIniciando...\n")
    sport_client.StandUp()
    time.sleep(2)
    sport_client.ClassicWalk(True)
    time.sleep(2)

    actual_pos = streaming_client.get_last_pos()
    objetive_pos = streaming_client.get_objetive_pos()

    while True:

        # Verificar si se presionó ESC
        if is_esc_pressed():
            print("\nSaliendo del bucle...")
            break

        obs_t = actual_pos  #Obtenemos las observaciones en T
        distanciaT = distancia(obs_t, objetive_pos)
        alfa_objT = alfa_obj(obs_t, objetive_pos)
        alfa_robotT = alfa_robot(obs_t)


        #action = random_action()  #Obtenemos una acción aleatoria

        
        user_interface.terminal_handle()
        action = test_option.id
        
        datos_entrada = np.concat([distanciaT, alfa_objT, alfa_robotT, action])
        datos_entrada_2 = np.concat([obs_t, objetive_pos, action])

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

        """

        #Esperar a que acabe el movimiento
        time.sleep(3)
        
        #Comprobar si el robot se ha salido de los límites establecidos
        if actual_pos[0][0] >= 2.0 or actual_pos[0][0] <= -1.1 or actual_pos[0][2] >= 1.25 or actual_pos[0][2] <= -1.4:
            out_of_limits()
            actual_pos = streaming_client.get_last_pos()
            continue
        else:
            actual_pos = streaming_client.get_last_pos()
        

        print("\n\n")
        print("Observaciones en T: x={obs_t[0][0]}, y={obs_t[0][1]}, z={obs_t[0][2]}\nx={obs_t[1][0]}, y={obs_t[1][1]}, z={obs_t[1][2]}\nx={obs_t[2][0]}, y={obs_t[2][1]}, z={obs_t[2][2]}")
        print(f"Observaciones en T: distancia={distanciaT}, alfa_obj={alfa_objT}, alfa_robot={alfa_robotT}")
        print(f"Acción tomada: {action}")
        print(f"Posición actual: x={actual_pos[0][0]}, y={actual_pos[0][1]}, z={actual_pos[0][2]}\nx={actual_pos[1][0]}, y={actual_pos[1][1]}, z={actual_pos[1][2]}\nx={actual_pos[2][0]}, y={actual_pos[2][1]}, z={actual_pos[2][2]}")
        print("\n\n")

        #Obtenemos las observaciones en T+1
        obs_t1 = actual_pos  
        
        distanciaT1 = distancia(obs_t1, objetive_pos)
        alfa_objT1 = alfa_obj(obs_t1, objetive_pos)
        alfa_robotT1 = alfa_robot(obs_t1)
        
        trainnig_data.append((datos_entrada, distanciaT1, alfa_objT1, alfa_robotT1))
        trainnig_data_2.append((datos_entrada_2, obs_t1, objetive_pos))
    
    sport_client.StopMove()
    time.sleep(2)
    sport_client.StandDown()
    time.sleep(3)

    print(f"Total de transiciones recolectadas: {len(trainnig_data)}")
    print(f"Transiciones recolectadas: {trainnig_data}")

    #Guardar los datos de entrenamiento en un archivo pickle
    with open("lista_obs.pkl", "wb") as f:
        pickle.dump(trainnig_data, f)

    with open("lista_obs_posicion_abs.pkl", "wb") as f:
        pickle.dump(trainnig_data_2, f)