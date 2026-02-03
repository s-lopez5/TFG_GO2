import time
import sys
import os
import numpy as np
import random
import pickle

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
    """
    return random.uniform(0, 1), random.uniform(0, 1)
    """

def random_action():
    """
    Genera una acción aleatoria para los ángulos alfa y beta.
    Las acciones son incrementos de entre -15 y 15 grados.
    """

    return random.randint(-15, 15), random.randint(-15, 15)

if __name__ == "__main__":
    WM_Data = []

    L = 0.2  #Longitud del brazo del robot
    x_obj, y_obj = pos_objetivo(L)
    print(f"Posición objetivo: ({x_obj}, {y_obj})")

    alfa, beta = random.randint(0, 90), random.randint(0, 90)  #Ángulos iniciales

    for i in range(1000):  #Simular 1000 pasos
        
        print(f"--- Iteración {i} ---")

        """
        #Cada 100 iteraciones, cambiar la posicion del objetivo
        if i % 100 == 0:
            x_obj, y_obj = pos_objetivo(L)
            print(f"Nuevo objetivo en: ({x_obj}, {y_obj})")
        """

        #Posicion en el paso t
        x_robot = pos_x2(L, alfa, beta)
        y_robot = pos_y2(L, alfa, beta)

        dist = distancia(x_obj, y_obj, x_robot, y_robot)
        angle_to_obj = alfa_obj(x_obj, y_obj, x_robot, y_robot)

        #Observación actual (distancia y ángulo al objetivo)
        obs_t = np.array([dist, angle_to_obj])  

        #Accion aleatoria
        delta_alfa, delta_beta = random_action()
        
        #Comprobar que los ángulos no se salgan de los límites
        if alfa + delta_alfa < 0 or alfa + delta_alfa > 90 or beta + delta_beta < 0 or beta + delta_beta > 90:
            continue
        
        #Datos de entrada para el paso t(distancia, ángulo al objetivo, cambios en alfa y beta)
        #datos_entrada = np.array([obs_t[0], obs_t[1], delta_alfa, delta_beta])
        datos_entrada = np.array([obs_t[0], obs_t[1], 0, delta_alfa, delta_beta])

        #Actualizar ángulos
        alfa += delta_alfa
        beta += delta_beta

        #Posicion en el paso t+1
        x_robot_2 = pos_x2(L, alfa, beta)
        y_robot_2 = pos_y2(L, alfa, beta)

        dist = distancia(x_robot_2, y_robot_2, x_obj, y_obj)
        angle_to_obj = alfa_obj(x_robot_2, y_robot_2, x_obj, y_obj)

        #Observación en el siguiente paso (distancia y ángulo al objetivo)
        #obs_t1 = np.array([dist, angle_to_obj])
        obs_t1 = np.array([dist, angle_to_obj, 0])

        print(f"Posición robot t: ({x_robot}, {y_robot})")
        print(f"Ángulos: (alfa: {alfa}º, beta: {beta}º)")
        print(f"Acción tomada: (delta_alfa: {delta_alfa}º, delta_beta: {delta_beta}º)")
        print(f"Posición robot t+1: ({x_robot_2}, {y_robot_2})")
        print(f"Distancia al objetivo: {dist}")
        print(f"Ángulo al objetivo: {np.degrees(angle_to_obj)} radianes\n")

        #Almacenar datos (observación en t, acción, observación en t+1)
        WM_Data.append((datos_entrada, obs_t1))

    print(f"Total de transiciones recolectadas: {len(WM_Data)}")
    print(f"Transiciones recolectadas: {WM_Data}")
    
    #Separar entradas y salidas
    inputs = np.array([item[0] for item in WM_Data])
    outputs = np.array([item[1] for item in WM_Data])

    #Guardar los datos en un archivo pickle
    with open("WM_data_Brazo_Robot.pkl", "wb") as f:
            pickle.dump({
            'inputs': inputs,
            'outputs': outputs
        }, f)