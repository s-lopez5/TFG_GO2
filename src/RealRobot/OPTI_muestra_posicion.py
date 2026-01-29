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

    actual_pos = []         #Posicion actual del robot
    objetive_pos = []       #Posicion del objetivo

    actual_pos = streaming_client.get_last_pos()

    while True:

        actual_pos = streaming_client.get_last_pos()
        objetive_pos = streaming_client.get_objetive_pos()  #Actualizar la posición del objetivo
        
        #Calcular observaciones actuales
        distanciaT = distancia(actual_pos, objetive_pos)
        alfa_objT = alfa_obj(actual_pos, objetive_pos)
        alfa_robotT = alfa_robot(actual_pos)

        print(f"Posición Actual del Robot: {actual_pos}")
        #print(f"Posición del Objetivo: {objetive_pos}")
        #print(f"Distancia: {distanciaT:.3f} m, Alfa Objetivo: {np.degrees(alfa_objT):.2f}°, Alfa Robot: {np.degrees(alfa_robotT):.2f}°")
        time.sleep(3)